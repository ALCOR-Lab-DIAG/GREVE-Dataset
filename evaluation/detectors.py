"""The detectors: one class each, written from the official test code (see adapters/).

Run each detector in its own environment (its repo's requirements): the repos pin incompatible
versions of torch, timm, CLIP and transformers, so they cannot share one.
"""

from __future__ import annotations

from base import Detector, register
from adapters.aigc import *       # noqa: F401,F403  CNNSpot ... PatchCraft (AIGCDetectBenchmark)
from adapters.modern_a import *   # noqa: F401,F403  DRCT, NPR, FatFormer
from adapters.modern_b import *   # noqa: F401,F403  RINE, CLIPDet, C2P-CLIP
from adapters.modern_c import *   # noqa: F401,F403  AIDE, SAFE, Effort
from adapters.modern_d import *   # noqa: F401,F403  B-Free, SPAI, Community Forensics
from adapters.modern_e import *   # noqa: F401,F403  GAPL, PGC, DGS-Net


@register
class Dummy(Detector):
    """Mean brightness as fake score. Only for testing the pipeline on the toy dataset."""
    name = "dummy"
    batch_size = 32

    def load(self, device):
        pass

    def preprocess(self, img):
        import numpy as np
        return np.asarray(img.resize((32, 32)), dtype=float).mean() / 255

    def forward(self, batch):
        return batch


@register
class MoEJEPA(Detector):
    """MoE-JEPA (AI4MFDD @ ECCV 2026). predict.py: load_moejepa(released .safetensors) on the frozen
    V-JEPA 2 ViT-L backbone, preprocess_image (static 4-frame clip), softmax over real / fake / tampered.
    Score = P(fake), the fully-synthetic class. Needs transformers >= 4.53: run it in the css-vlm image."""
    name = "moe_jepa"
    code = "https://github.com/ALCOR-Lab-DIAG/MoE-JEPA"
    weights = [("url", "https://github.com/ALCOR-Lab-DIAG/MoE-JEPA/releases/download/v1.0/moejepa_sidset.safetensors",
                "moejepa_sidset.safetensors")]
    checkpoint = "moejepa_sidset.safetensors (release v1.0, trained on SID-Set; classes real / fake / tampered)"
    preprocessing = ("V-JEPA 2 video processor of facebook/vjepa2-vitl-fpc64-256 on the image, repeated as a static "
                     "4-frame clip (model/model.py:preprocess_image)")
    batch_size = 16

    def load(self, device):
        self.use_repo()
        import sys
        for m in list(sys.modules):
            if m.split(".")[0] == "model":
                del sys.modules[m]
        from model.model import load_moejepa, load_processor, preprocess_image
        self.model = load_moejepa(str(self.home / "moejepa_sidset.safetensors"), device=device)
        self.processor, self._prep, self.device = load_processor(), preprocess_image, device

    def preprocess(self, img):
        return self._prep(img, self.processor)

    def forward(self, batch):
        import torch
        with torch.no_grad():
            return self.model(torch.stack(batch).to(self.device)).float().softmax(-1)[:, 1].tolist()
