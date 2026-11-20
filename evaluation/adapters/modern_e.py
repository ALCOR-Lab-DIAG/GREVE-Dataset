from base import IMAGENET, Detector, find, register, sigmoid_scores, to_tensor


def _forget(*names):
    """Drop cached modules with generic names (models, data, utils) imported from another repo."""
    import sys
    for m in list(sys.modules):
        if m.split(".")[0] in names:
            del sys.modules[m]


@register
class GAPL(Detector):
    """GAPL (CVPR 2026). inference_image.py / validate_bench.py: GAPLModel(freeze_backbone=False) = CLIP ViT-L/14
    vision tower with LoRA on q/k/v, checkpoint['model'] loaded with strict=False plus checkpoint['prototype']."""
    name = "gapl"
    code = "https://github.com/UltraCapture/GAPL"
    weights = [("hf_file", "AbyssLumine/GAPL", "checkpoint.pt", "checkpoint.pt")]
    checkpoint = "AbyssLumine/GAPL checkpoint.pt (stage 2, the one of the README's inference and val_bench)"
    preprocessing = "no resize; CenterCrop 224, ImageNet norm (inference_image.py, validate_bench.py stat 'imagenet')"
    batch_size = 32

    def load(self, device):
        import torch
        from huggingface_hub import snapshot_download
        # models.py loads CLIP from this exact snapshot of the HF cache, local_files_only
        snapshot_download("openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41")
        self.use_repo()
        _forget("models")
        import models
        model = models.GAPLModel(fe_path=None, proto_path=None, freeze_backbone=False, device=device)
        ck = torch.load(self.home / "checkpoint.pt", map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"gapl: unexpected keys {unexpected[:5]}"
        model.load_prototype(ck["prototype"])
        self.model, self.device = model.to(device).eval(), device

    def preprocess(self, img):
        return to_tensor(img, IMAGENET, crop=224)

    def forward(self, batch):
        return sigmoid_scores(self.model, batch, self.device)


@register
class PGC(Detector):
    """PGC (ICML 2026). test.py: PGCNetwork(dinov2-large, LoRA r=8 alpha=1 on attn.qkv/proj, mlp.fc1/fc2,
    tau 0.5/0.5), state['model'] strict; score = sigmoid of the calibrated logit (engine/evaluator.py)."""
    name = "pgc"
    code = "https://github.com/xiaoyu6868/PGC"
    weights = [("hf_file", "xiaoyuzhou68/PGC_ckpt", "PGC_train_sdv1_4_ckpt.pth", "PGC_train_sdv1_4_ckpt.pth"),
               ("hf", "facebook/dinov2-large", "pretrained_dino/dinov2-large")]
    checkpoint = ("PGC_train_sdv1_4_ckpt.pth (trained on GenImage SDv1.4; the README uses it for GenImage and for "
                  "CommGen15, the paper's benchmark of commercial generators)")
    preprocessing = ("no resize; PadCenterCrop 224 (pad if smaller), ToTensor, + quantization residual as 3 extra "
                     "channels, ImageNet norm on RGB and 0/0.5 on the residual (data/transforms.py:create_eval_transforms)")
    batch_size = 32

    def load(self, device):
        import torch
        self.use_repo()
        _forget("models", "data", "utils", "engine")
        from data.transforms import MEAN, STD, create_eval_transforms
        from models.pgc import PGCNetwork
        self.transform = create_eval_transforms(image_size=224, mean=MEAN["dino"], std=STD["dino"])
        model = PGCNetwork(dino_variant="dinov2-large", lora_rank=8, lora_alpha=1.0, lora_dropout=0.1,
                           lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
                           pretrained_root=str(self.home / "pretrained_dino"), tau_rgb=0.5, tau_res=0.5)
        state = torch.load(self.home / "PGC_train_sdv1_4_ckpt.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        self.model, self.device = model.to(device).eval(), device

    def preprocess(self, img):
        return self.transform(img)

    def forward(self, batch):
        import torch
        with torch.no_grad():
            _, logits = self.model(torch.stack(batch).to(self.device), return_feature=True)
        return torch.sigmoid(logits.view(-1)).tolist()


@register
class DGSNet(Detector):
    """DGS-Net (ICML 2026). test.py: CLIPModel(opt) = CLIP ViT-L/14 (OpenAI clip) with LoRA (q,k,v, all layers,
    r=6, alpha=6), loaded strict from model_epoch_step2.pth; it also builds a frozen copy from
    ./checkpoints/model_epoch_step1.pth (relative path), unused at test time. Score = sigmoid(img_logits)."""
    name = "dgs_net"
    code = "https://github.com/HorizonTEL/DGS-Net"
    weights = [("gdrive_folder", "1_MznDWLcZDNCeWj6AeP8MrJBLoMomxe7", "checkpoints")]
    checkpoint = "model_epoch_step2.pth (README test command; trained on AIGIBench ProGAN + SDv1.4)"
    preprocessing = ("no resize; 7x7 grid of 32x32 patches, the 24 lowest and 25 highest spectral-entropy ones, "
                     "shuffled (python random, seeded by detect.py --seed) and stitched to 224x224; ImageNet norm "
                     "(data/PatchSelectionTransform.py, data/datasets.py)")
    batch_size = 32

    def load(self, device):
        import os
        import sys
        import torch
        import torchvision.transforms as T
        self.use_repo()
        _forget("models", "data", "options", "util", "networks")
        from data.PatchSelectionTransform import PatchSelectionTransform
        from models.clip_models import CLIPModel
        from options.test_options import TestOptions
        argv, sys.argv = sys.argv, ["test.py", "--model_path", "unused", "--gpu_ids", "-1"]
        try:
            opt = TestOptions().parse(print_options=False)
        finally:
            sys.argv = argv
        ckpts = find(self.home / "checkpoints", "model_epoch_step2.pth").parent
        os.chdir(ckpts.parent)  # CLIPModel reads ./checkpoints/model_epoch_step1.pth
        if ckpts.name != "checkpoints":
            raise FileNotFoundError(f"dgs_net: expected the weights in a 'checkpoints' folder, found {ckpts}")
        load = torch.load  # the frozen copy is torch.load-ed without map_location and was saved on cuda:1
        torch.load = lambda *a, **k: load(*a, **{"map_location": "cpu", **k})
        try:
            model = CLIPModel(opt)
        finally:
            torch.load = load
        model.load_state_dict(torch.load(ckpts / "model_epoch_step2.pth", map_location="cpu"), strict=True)
        self.transform = T.Compose([PatchSelectionTransform(patch_size=32, num_patches=49), T.ToTensor(),
                                    T.Normalize(*IMAGENET)])
        self.model, self.device = model.to(device).eval(), device

    def preprocess(self, img):
        return self.transform(img)

    def forward(self, batch):
        import torch
        with torch.no_grad():
            return self.model(torch.stack(batch).to(self.device))[0].sigmoid().flatten().tolist()
