from base import IMAGENET, Detector, find, register, to_tensor


def _forget(*names):
    """Drop cached modules with generic names (networks, utils, models) imported from another repo."""
    import sys
    for m in list(sys.modules):
        if m.split(".")[0] in names:
            del sys.modules[m]


@register
class BFree(Detector):
    name = "bfree"
    code = "https://github.com/grip-unina/B-Free"
    weights = [("url", "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip",
                "BFREE_dino2reg4.zip")]
    checkpoint = "BFREE_dino2reg4/model_epoch_best.pth (the only released model)"
    preprocessing = ("no resize; ToTensor + norm_type of config.yaml; the network averages 5 crops "
                     "of 504x504 (center + 4 corners) taken on patch embeddings, smaller images tiled")
    threshold = 0.5  # sigmoid(logit) > 0.5, i.e. logit > 0 as in main_bfree.py:compute_metrics (bAcc)
    batch_size = 1   # variable-size inputs, main_bfree.py runs one image at a time

    def load(self, device):
        import sys
        import torchvision.transforms as T
        import yaml
        self.use_repo()
        sys.path.insert(0, str(self.repo / "code"))
        _forget("networks", "utils")
        from networks import get_network, load_weights
        from utils.normalization import get_list_norm
        # code/main_bfree.py:get_config / running_tests
        cfg_file = find(self.home, "config.yaml")
        cfg = yaml.load(cfg_file.read_text(), Loader=yaml.FullLoader)
        self.model = load_weights(get_network(cfg["arch"]), str(cfg_file.parent / cfg["weights_file"]))
        self.model = self.model.to(device).eval()
        self.transform = T.Compose(get_list_norm(cfg["norm_type"]))
        self.device = device

    def preprocess(self, img):
        return self.transform(img)

    def forward(self, batch):
        import torch
        out = []
        with torch.no_grad():
            for x in batch:
                y = self.model(x.unsqueeze(0).to(self.device))
                # one logit, or fake-minus-real for two logits (main_bfree.py)
                y = y[:, 0] if y.shape[1] == 1 else y[:, 1] - y[:, 0]
                out.append(y.sigmoid().item())
        return out


@register
class SPAI(Detector):
    name = "spai"
    code = "https://github.com/mever-team/spai"
    weights = [("gdrive", "1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI", "spai.pth")]
    checkpoint = "spai.pth (the only released model, README 'Weights Checkpoint')"
    preprocessing = ("original resolution, reflect-101 pad to at least 224x224, /255 (no mean/std); "
                     "the model splits into 224 patches (stride 224, >=4 patches else five-crop)")
    threshold = 0.5  # sigmoid output, spai/metrics.py Metrics(threshold=0.5); class 1 = fake
    batch_size = 1   # variable-size inputs, `spai infer` default --batch-size 1

    def load(self, device):
        import logging
        self.use_repo()
        from spai.config import get_config
        from spai.models import build_cls_model
        from spai.utils import load_pretrained
        # spai/__main__.py:infer with its defaults (--cfg configs/spai.yaml)
        self.config = get_config({"cfg": str(self.repo / "configs" / "spai.yaml"), "opts": ()})
        self.model = build_cls_model(self.config)
        load_pretrained(self.config, self.model, logging.getLogger("spai"),
                        checkpoint_path=find(self.home, "spai.pth"), verbose=False)
        self.model = self.model.to(device).eval()
        self.device = device

    def preprocess(self, img):
        # test-time transform of spai/data/data_finetune.py:build_transform with TEST.ORIGINAL_RESOLUTION:
        # A.PadIfNeeded(224, 224) (centered, cv2.BORDER_REFLECT_101) + A.Normalize(mean=0, std=1) + ToTensorV2
        import numpy as np
        import torch
        x = np.asarray(img, dtype=np.float32) / 255.0
        s = self.config.DATA.IMG_SIZE
        ph, pw = max(s - x.shape[0], 0), max(s - x.shape[1], 0)
        if ph or pw:
            x = np.pad(x, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)), mode="reflect")
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def forward(self, batch):
        import torch
        with torch.no_grad():
            # arbitrary-resolution forward: list of 1xCxHxW tensors (spai/__main__.py:validate)
            out = self.model([x.unsqueeze(0).to(self.device) for x in batch],
                             self.config.MODEL.FEATURE_EXTRACTION_BATCH)
            return torch.sigmoid(out).flatten().tolist()


@register
class CommunityForensics(Detector):
    name = "community_forensics"
    code = "https://github.com/JeongsooP/Community-Forensics"
    weights = [("hf", "OwensLab/commfor-model-384", "hf")]
    checkpoint = "OwensLab/commfor-model-384 (ViT-S/16, 384 input; README's main HF checkpoint)"
    preprocessing = "Resize(440) bilinear, CenterCrop(384), ToTensor, ImageNet norm"
    threshold = 0.5  # sigmoid output, torchmetrics BinaryAccuracy default threshold in utils.py
    batch_size = 32

    def load(self, device):
        self.use_repo()
        _forget("models", "utils")
        import models
        # eval_using_huggingface.ipynb: models.ViTClassifier.from_pretrained('OwensLab/commfor-model-384')
        self.model = models.ViTClassifier.from_pretrained(str(self.home / "hf"), device=device)
        self.model = self.model.to(device).eval()
        self.device = device

    def preprocess(self, img):
        # test-time transform of dataloader.py:get_transform (mode="test", input_size=384)
        return to_tensor(img, IMAGENET, size=440, crop=384)

    def forward(self, batch):
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.stack(batch).to(self.device))).flatten().tolist()
