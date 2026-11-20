from base import CLIP, Detector, find, register, sigmoid_scores, to_tensor


@register
class RINE(Detector):
    name = "rine"
    code = "https://github.com/mever-team/rine"
    weights = []  # ckpt/*.pth are plain files in the repo
    checkpoint = "ckpt/model_4class_trainable.pth (ProGAN 4-class; the one used by demo/demo.py)"
    preprocessing = "CenterCrop 224 (no resize), CLIP normalisation (src/utils.py:get_transforms, transforms_test_1)"
    threshold = 0.5
    batch_size = 32

    def load(self, device):
        import torch
        self.use_repo()
        from src.models import Model
        self.device = device
        # hyperparameters of the 4-class model, src/utils.py:get_our_trained_model(ncls=4)
        self.model = Model(backbone=("ViT-L/14", 1024), nproj=2, proj_dim=1024, device=device)
        sd = torch.load(self.repo / "ckpt" / "model_4class_trainable.pth", map_location="cpu")
        # the official loader assigns each tensor by name (trainable part only); same as this
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("clip.") for k in missing), (missing, unexpected)
        self.model.to(device).eval()

    def preprocess(self, img):
        return to_tensor(img, CLIP, crop=224)

    def forward(self, batch):
        import torch
        with torch.no_grad():
            logit = self.model(torch.stack(batch).to(self.device))[0]  # Model returns (p, z)
        return logit.sigmoid().flatten().tolist()


@register
class ClipDet(Detector):
    name = "clipdet"
    code = "https://github.com/grip-unina/ClipBased-SyntheticImageDetection"
    weights = [("lfs",)]
    # CLIP detector alone. main.py's default also fuses Corvi2023 (a ResNet-50): add it to `models` for that.
    checkpoint = "weights/clipdet_latent10k_plus"
    preprocessing = "Resize 224 bicubic + CenterCrop 224, CLIP norm (main.py:runnig_tests, config.yaml)"
    threshold = 0.0  # LLR, README: "If LLR>0, the image is detected as synthetic"
    batch_size = 1

    models = ("clipdet_latent10k_plus",)

    def load(self, device):
        import yaml
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
        self.use_repo()
        from networks import create_architecture, load_weights
        from utils.processing import make_normalize
        self.device = device
        self.nets, self.tfs = [], []
        for m in self.models:  # main.py:get_config + runnig_tests
            cfg = yaml.safe_load(open(self.repo / "weights" / m / "config.yaml"))
            net = load_weights(create_architecture(cfg["arch"]), str(self.repo / "weights" / m / cfg["weights_file"]))
            self.nets.append(net.to(device).eval())
            ps = cfg["patch_size"]
            if ps is None:
                t = []
            elif ps == "Clip224":
                t = [Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop((224, 224))]
            else:
                raise NotImplementedError(f"patch_size {ps!r}")
            self.tfs.append(Compose(t + [make_normalize(cfg["norm_type"])]))

    def preprocess(self, img):
        return [t(img) for t in self.tfs]

    def forward(self, batch):
        import numpy as np
        import torch
        from utils.fusion import apply_fusion
        scores = []
        with torch.no_grad():
            for x in batch:
                logits = []
                for net, t in zip(self.nets, x):
                    out = net(t[None].to(self.device)).cpu().numpy()
                    out = out[:, 0] if out.shape[1] == 1 else out[:, 1] - out[:, 0]  # as in main.py
                    if out.ndim > 1:
                        out = out.mean((1, 2))
                    logits.append(float(out[0]))
                fused = logits[0] if len(logits) == 1 else apply_fusion(np.array([logits]), "soft_or_prob", axis=-1)[0]
                scores.append(float(fused))
        return scores


@register
class C2PCLIP(Detector):
    name = "c2p_clip"
    code = "https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection"
    weights = [("gdrive_folder", "1WZStlW2zpH85NZit1-JADMvzovEEZEj9", "drive")]
    checkpoint = ("C2P_CLIP_release_20240901 (UniversalFakeDetect / ProGAN 4-class; default --model_path "
                  "of scripts/inference.py)")
    preprocessing = ("Resize 224 (bilinear) + CenterCrop 224, CLIP norm (data.py:dataset_folder of the "
                     "release commit 27b5232 that scripts/inference.py was written for)")
    threshold = 0.5
    batch_size = 32

    def load(self, device):
        import zipfile
        import torch
        import torch.nn as nn
        from transformers import CLIPModel
        self.device = device

        # C2P_CLIP class copied from scripts/inference.py (that script cannot be imported: it parses args
        # and imports `data` at module level)
        class C2P_CLIP(nn.Module):
            def __init__(self, name="openai/clip-vit-large-patch14", num_classes=1):
                super().__init__()
                self.model = CLIPModel.from_pretrained(name)
                del self.model.text_model
                del self.model.text_projection
                del self.model.logit_scale
                self.model.fc = nn.Linear(768, num_classes)

            def forward(self, img):
                out = self.model.vision_model(pixel_values=img, return_dict=True)
                emb = self.model.visual_projection(out[1])
                return self.model.fc(emb / emb.norm(p=2, dim=-1, keepdim=True))

        try:
            p = find(self.home / "drive", "C2P_CLIP_release_20240901*")
        except FileNotFoundError:
            files = sorted(str(f.relative_to(self.home)) for f in self.home.rglob("*") if f.is_file())
            raise FileNotFoundError(f"c2p_clip: C2P_CLIP_release_20240901* not in the Drive folder; found {files}")
        if p.suffix == ".zip":  # torch.hub legacy zip (as load_state_dict_from_url): one checkpoint inside
            with zipfile.ZipFile(p) as z:
                z.extractall(p.with_suffix(""))
            p = p.with_suffix("")
        if p.is_dir():
            members = [f for f in p.rglob("*") if f.is_file()]
            assert len(members) == 1, f"c2p_clip: expected one checkpoint file in {p}, got {members}"
            p = members[0]
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if "model" in sd and isinstance(sd["model"], dict):
            raise RuntimeError("c2p_clip: this is a training-code (LoRA Trainer) checkpoint, not the release "
                               "state dict expected by scripts/inference.py")
        self.model = C2P_CLIP()
        self.model.load_state_dict(sd, strict=True)  # strict, as in scripts/inference.py
        self.model.to(device).eval()

    def preprocess(self, img):
        return to_tensor(img, CLIP, size=224, crop=224)

    def forward(self, batch):
        return sigmoid_scores(self.model, batch, self.device)
