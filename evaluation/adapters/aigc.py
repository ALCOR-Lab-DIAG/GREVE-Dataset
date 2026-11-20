from base import CACHE, Detector, find, register, sigmoid_scores, state_dict

import os
import sys
from contextlib import contextmanager


@contextmanager
def _argv_cwd(argv, cwd):
    """Run repo code that reads sys.argv / relative paths without touching the real ones."""
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = [old_argv[0] if old_argv else "x"] + list(argv)
    os.chdir(cwd)
    try:
        yield
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


class _AIGC(Detector):
    """AIGCDetectBenchmark (eval_all.py): one repo and one weights folder shared by all its methods."""
    code = "https://github.com/Ekko-zn/AIGCDetectBenchmark"
    weights = [("gdrive_folder", "1Pqva_aOFU36wAidspj3P0FK34CbgaZUQ", "weights")]  # classifier/ + preprocessing/
    threshold = 0.5  # validate.py: sigmoid > 0.5
    batch_size = 32  # every method except Fusing returns a fixed 224x224 tensor
    method = ""      # --detect_method
    ckpt = ""        # file in weights/classifier
    state_key = "model"  # eval_all.py: FreDect/Gram 'netC', UnivFD bare fc, others 'model'

    @property
    def home(self):
        return CACHE / "aigcbench"

    def load(self, device):
        import torch
        self.use_repo()
        from options import TestOptions
        from util import get_model
        from data.process import get_processing_model
        from data.datasets import custom_augment, process_img
        self.custom_augment, self.process_img = custom_augment, process_img
        self.device = device
        w = self.home / "weights"
        args = ["--detect_method", self.method, "--model_path", str(find(w, self.ckpt))]
        if self.method == "LNP":
            args += ["--LNP_modelpath", str(find(w, "sidd_rgb.pth"))]
        if self.method == "DIRE":
            args += ["--DIRE_modelpath", str(find(w, "lsun_bedroom.pt"))]
        if self.method == "LGrad":
            args += ["--LGrad_modelpath", str(find(w, "karras2019stylegan-bedrooms-256x256_discriminator.pth"))]
        # TestOptions defaults (eval_all.py, no flags): loadSize 256, CropSize 224, resize+crop on, noise_type None
        with _argv_cwd(args, self.repo):
            opt = TestOptions().parse(print_options=False)
        opt.process_device = torch.device(device)
        # get_processing_model reads ./weights/auxiliary (FreDect) and sys.argv (DIRE create_argparser)
        with _argv_cwd([], self.repo):
            opt = get_processing_model(opt)
        if self.method == "LNP":  # processing_LNP never moves the input to process_device
            net = opt.model_restoration
            opt.model_restoration = lambda x: net(x.to(opt.process_device))
        self.opt = opt
        model = get_model(opt)
        if self.method == "UnivFD":
            model.fc.load_state_dict(state_dict(opt.model_path, keys=()))
        else:
            model.load_state_dict(state_dict(opt.model_path, keys=(self.state_key,)), strict=True)
        self.model = model.to(device).eval()

    def preprocess(self, img):
        import torch
        img = self.custom_augment(img, self.opt)  # identity for noise_type None
        if self.method == "LGrad":  # needs gradients of the discriminator
            return self.process_img(img, self.opt, "", 0)[0]
        with torch.no_grad():
            return self.process_img(img, self.opt, "", 0)[0]

    def forward(self, batch):
        return sigmoid_scores(self.model, batch, self.device)


@register
class CNNSpot(_AIGC):
    name, method, ckpt = "cnnspot", "CNNSpot", "CNNSpot.pth"
    checkpoint = "weights/classifier/CNNSpot.pth (AIGCDetectBenchmark release)"
    preprocessing = "Resize 256 (short side, bilinear) + CenterCrop 224, ImageNet norm (data/process.py:processing)"


@register
class FreDect(_AIGC):
    name, method, ckpt, state_key = "fredect", "FreDect", "DCTAnalysis.pth", "netC"
    checkpoint = "weights/classifier/DCTAnalysis.pth (AIGCDetectBenchmark release)"
    preprocessing = ("Resize 256 + CenterCrop 224, log|DCT| normalised with weights/auxiliary/dct_mean, dct_var "
                     "(data/process.py:processing_DCT)")


@register
class Fusing(_AIGC):
    name, method, ckpt = "fusing", "Fusing", "PSM.pth"
    checkpoint = "weights/classifier/PSM.pth (AIGCDetectBenchmark release)"
    preprocessing = ("full image + Resize 224/CenterCrop 224 view + (H, W), ImageNet norm "
                     "(data/process.py:processing_PSM)")
    batch_size = 1  # the full-resolution image varies in size

    def preprocess(self, img):
        img = self.custom_augment(img, self.opt)
        input_img, cropped_img, _, scale, _ = self.process_img(img, self.opt, "", 0)
        return input_img, cropped_img, scale

    def forward(self, batch):
        import torch
        # validate.py:validate_PSM: input_img stays a list on CPU, cropped_img and scale on the GPU
        with torch.no_grad():
            logits = self.model([b[0] for b in batch], torch.stack([b[1] for b in batch]).to(self.device),
                                torch.stack([b[2] for b in batch]).to(self.device))
        return logits.sigmoid().flatten().tolist()


@register
class GramNet(_AIGC):
    name, method, ckpt, state_key = "gramnet", "Gram", "Gram.pth", "netC"
    checkpoint = "weights/classifier/Gram.pth (AIGCDetectBenchmark release)"
    preprocessing = "Resize 256 + CenterCrop 224, ImageNet norm (data/process.py:processing)"


@register
class LGrad(_AIGC):
    name, method, ckpt = "lgrad", "LGrad", "LGrad.pth"
    checkpoint = ("weights/classifier/LGrad.pth + preprocessing/"
                  "karras2019stylegan-bedrooms-256x256_discriminator.pth (AIGCDetectBenchmark release)")
    preprocessing = ("Resize 256x256, [-1,1]; gradient of the StyleGAN-bedroom discriminator, min-max to "
                     "uint8 PNG; then Resize 256 + CenterCrop 224, ImageNet norm (data/process.py:processing_LGrad)")


@register
class LNP(_AIGC):
    name, method, ckpt = "lnp", "LNP", "LNP.pth"
    checkpoint = "weights/classifier/LNP.pth + preprocessing/sidd_rgb.pth (AIGCDetectBenchmark release)"
    preprocessing = ("SIDD denoiser (CycleISP DenoiseNet) on the full image, output as in processing_LNP; then "
                     "Resize 256 + CenterCrop 224, ImageNet norm (data/process.py:processing_LNP)")


@register
class DIRE(_AIGC):
    name, method, ckpt = "dire", "DIRE", "DIRE.pth"
    checkpoint = "weights/classifier/DIRE.pth + preprocessing/lsun_bedroom.pt ADM (AIGCDetectBenchmark release)"
    preprocessing = ("center_crop_arr 256, DDIM-20 inversion + reconstruction with ADM lsun_bedroom (fp32), "
                     "|x - recon| as uint8; then Resize 256 + CenterCrop 224, ImageNet norm "
                     "(data/process.py:processing_DIRE, util.create_argparser)")


@register
class UnivFD(_AIGC):
    name, method, ckpt = "univfd", "UnivFD", "UnivFD.pth"
    checkpoint = "weights/classifier/UnivFD.pth (linear head on CLIP ViT-L/14, AIGCDetectBenchmark release)"
    preprocessing = "Resize 256 (short side, bilinear) + CenterCrop 224, CLIP norm (data/process.py:processing)"


@register
class PatchCraft(_AIGC):
    """RPTC. Official test code is only on ModelScope (aemilia/AIGCDetectionBenchmark, unreachable); this uses
    the SIDBench (mever-team) implementation, which loads the same RPTC.pth."""
    name, ckpt = "patchcraft", "RPTC.pth"
    code = "https://github.com/mever-team/sidbench"
    checkpoint = "weights/classifier/RPTC.pth (AIGCDetectBenchmark release), loaded with SIDBench models/RPTC.py"
    preprocessing = ("no resize; 192 random 28x28 crops sorted by texture (ED), 8x8 poor/rich mosaics of 224, "
                     "ToTensor only (SIDBench dataset/process.py:rptc_processing, patchNum 3, cropSize 224)")

    @property
    def repo(self):
        return self.home / "sidbench"

    def load(self, device):
        import types
        from argparse import Namespace
        self.use_repo()
        # stub the `models` package: its __init__ imports every SIDBench model (BLIP, CLIP, ...)
        pkg = types.ModuleType("models")
        pkg.__path__, pkg.VALID_MODELS = [str(self.repo / "models")], ["RPTC"]
        sys.modules["models"] = pkg
        from models.RPTC import Net
        from dataset.process import rptc_processing
        self.rptc_processing = rptc_processing
        self.opt = Namespace(patchNum=3, cropSize=224)  # options/options.py:TestOptions defaults
        self.device = device
        model = Net()
        model.load_weights(str(find(self.home / "weights", self.ckpt)))  # netC / plain / 'module.' fallback
        self.model = model.to(device).eval()

    def preprocess(self, img):
        import torch
        torch.manual_seed(self.seed)  # random crops; fixed per image so scores do not depend on order
        return self.rptc_processing(img, self.opt)
