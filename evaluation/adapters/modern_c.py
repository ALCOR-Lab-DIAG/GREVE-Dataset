from base import CLIP, IMAGENET, Detector, find, register, softmax_fake, state_dict, to_tensor

import os
import sys


def _forget(*mods):
    """Drop cached top-level packages these repos share (models, data, utils, ...), so each repo imports its own."""
    for m in list(sys.modules):
        if m.split(".")[0] in mods:
            del sys.modules[m]


@register
class AIDE(Detector):
    name = "aide"
    code = "https://github.com/shilinyan99/AIDE"
    # README "Model Zoo" Drive folder (several checkpoints, file names not listed in the repo)
    weights = [("gdrive_folder", "1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4", "ckpts")]
    ckpt_pattern = "progan"
    checkpoint = "AIDE trained on ProGAN (CNNSpot train set), the README's eval example progan_train.pth"
    preprocessing = ("ToTensor on the full image; DCT patch selection (32x32 windows, stride 16) -> min/max "
                     "patches; patches and full image Resize([256,256]) + ImageNet norm")
    threshold = 0.5
    batch_size = 16

    def load(self, device):
        import types
        _forget("models", "data", "utils")
        self.use_repo()
        sys.modules.setdefault("clip", types.ModuleType("clip"))  # models/AIDE.py imports openai `clip` but never uses it
        import models.AIDE as AIDE
        from data.dct import DCT_base_Rec_Module
        hits = [p for p in sorted(self.home.joinpath("ckpts").rglob("*.pth")) if self.ckpt_pattern in p.name.lower()]
        if not hits:
            raise FileNotFoundError(f"aide: no *{self.ckpt_pattern}*.pth among "
                                    f"{[p.name for p in self.home.joinpath('ckpts').rglob('*')]}")
        # scripts/eval.sh passes no --resnet_path/--convnext_path: the checkpoint holds the whole model
        # (utils.save_model saves model_without_ddp.state_dict(), incl. the frozen ConvNeXt-XXL)
        self.model = AIDE.AIDE(resnet_path=None, convnext_path=None)
        self.model.load_state_dict(state_dict(hits[0], keys=("model",)))  # utils.auto_load_model
        self.model.to(device).eval()
        self.device = device
        self.dct = DCT_base_Rec_Module()

    def preprocess(self, img):
        import torch
        import torchvision.transforms as T
        # data/datasets.py:TestDataset.__getitem__ (transform_before_test, dct, transform_train)
        # antialias=False: official envs (torchvision 0.12 / 0.15) do not antialias tensor resizes
        tf = T.Compose([T.Resize([256, 256], antialias=False), T.Normalize(*IMAGENET)])
        x = T.ToTensor()(img)
        with torch.no_grad():
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(x)
        return torch.stack([tf(x_minmin), tf(x_maxmax), tf(x_minmin1), tf(x_maxmax1), tf(x)], dim=0)

    def forward(self, batch):
        # engine_finetune.py:evaluate -> softmax(...)[:, 1] > 0.5
        return softmax_fake(self.model, batch, self.device)


@register
class SAFE(Detector):
    name = "safe"
    code = "https://github.com/Ouxiang-Li/SAFE"
    weights = []  # checkpoint/checkpoint-best.pth is a regular file in the repo (not LFS)
    checkpoint = "checkpoint/checkpoint-best.pth (ProGAN 4-class train set, the only released one)"
    preprocessing = "CenterCrop 256 (no resize), ToTensor, no normalisation; DWT inside the model"
    threshold = 0.5
    batch_size = 64

    def load(self, device):
        _forget("models", "data", "utils")
        self.use_repo()
        from models.resnet import resnet50
        self.model = resnet50(num_classes=2)  # main_finetune.py, --model SAFE
        self.model.load_state_dict(state_dict(find(self.repo, "checkpoint-best.pth"), keys=("model",)))
        self.model.to(device).eval()
        self.device = device

    def preprocess(self, img):
        # data/datasets.py:Get_Transforms, eval of --transform_mode crop --input_size 256 (scripts/eval.sh)
        import torchvision.transforms as T
        return T.Compose([T.CenterCrop([256, 256]), T.ToTensor()])(img)

    def forward(self, batch):
        # engine_finetune.py:evaluate -> softmax(...)[:, 1] > 0.5
        return softmax_fake(self.model, batch, self.device)


@register
class Effort(Detector):
    name = "effort"
    code = "https://github.com/YZY-stack/Effort-AIGI-Detection"
    clip_dir = "models--openai--clip-vit-large-patch14"  # effort_detector.py loads "../" + this, relative to cwd
    weights = [
        ("gdrive", "1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d", "effort_genimage_sdv14.pth"),  # README: trained on GenImage (sdv1.4)
        ("hf_file", "openai/clip-vit-large-patch14", "config.json", f"{clip_dir}/config.json"),
        ("hf_file", "openai/clip-vit-large-patch14", "model.safetensors", f"{clip_dir}/model.safetensors"),
    ]
    checkpoint = "CLIP ViT-L/14 + Effort trained on GenImage SDv1.4 (README AIGI checkpoint)"
    preprocessing = "cv2 resize to 224x224 (INTER_LINEAR, no crop), ToTensor, CLIP norm (training/demo.py)"
    threshold = 0.5
    batch_size = 32

    def load(self, device):
        _forget("utils", "detectors", "networks", "loss", "metrics")
        sys.path.insert(0, str(self.repo / "DeepfakeBench" / "training"))
        from detectors.effort_detector import EffortDetector
        cwd = os.getcwd()
        os.chdir(self.repo)  # so "../models--openai--clip-vit-large-patch14" is self.home/clip_dir
        try:
            self.model = EffortDetector({})  # config is only stored
        finally:
            os.chdir(cwd)
        # training/demo.py:load_detector (state_dict or plain dict, "module." stripped, strict=False)
        missing, unexpected = self.model.load_state_dict(
            state_dict(self.home / "effort_genimage_sdv14.pth", keys=("state_dict",)), strict=False)
        missing = [k for k in missing if not k.endswith(('.U_r', '.S_r', '.V_r'))]  # training-only (orthogonal loss)
        if missing or unexpected:
            print(f"effort: {len(missing)} missing, {len(unexpected)} unexpected keys, e.g. "
                  f"{missing[:3]} {unexpected[:3]}", flush=True)
        if any(k.startswith("head.") for k in missing) or len(missing) > 10:
            raise RuntimeError(f"effort: checkpoint does not match EffortDetector, missing {missing[:10]}")
        self.model.to(device).eval()
        self.device = device

    def preprocess(self, img):
        # training/demo.py:preprocess_face (image read with cv2 and converted to RGB = PIL RGB array)
        import cv2
        import numpy as np
        from PIL import Image
        x = cv2.resize(np.asarray(img), (224, 224), interpolation=cv2.INTER_LINEAR)
        return to_tensor(Image.fromarray(x), CLIP)

    def forward(self, batch):
        import torch
        # EffortDetector.forward: prob = softmax(head(pooler_output))[:, 1]
        with torch.no_grad():
            out = self.model({"image": torch.stack(batch).to(self.device)}, inference=True)
        return out["prob"].flatten().tolist()
