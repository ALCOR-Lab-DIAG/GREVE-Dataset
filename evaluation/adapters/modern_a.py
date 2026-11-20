from base import IMAGENET, Detector, find, register, sigmoid_scores, softmax_fake, state_dict, to_tensor


@register
class DRCTConvB(Detector):
    name = "drct_convb"
    code = "https://github.com/beibuwandeluori/DRCT"
    weights = [("modelscope_dataset", "BokingChen/DRCT-2M", "pretrained.zip", "pretrained.zip")]  # README: modelscope.cn/datasets/BokingChen/DRCT-2M
    checkpoint = "DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth (ConvNeXt-B + DRCT, trained on SD v1.4)"
    preprocessing = "albumentations: PadIfNeeded(224, zeros) + CenterCrop 224 + ImageNet Normalize (create_val_transforms is_crop=True)"
    threshold = 0.5
    batch_size = 32

    def load(self, device):
        self.use_repo()
        import numpy as np
        from data.transform import create_val_transforms
        from network.models import get_models
        # scripts/test_DRCT-2M.sh -> train.py --is_test: convnext_base_in22k, embedding_size 1024 (ContrastiveModels);
        # pretrained=False: no timm ImageNet download, everything comes from the released checkpoint
        self.model = get_models(model_name="convnext_base_in22k", num_classes=2, pretrained=False,
                                embedding_size=1024)
        ckpt = find(self.home / "pretrained",
                    "DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth")
        self.model.load_state_dict(state_dict(ckpt), strict=True)
        self.model.to(device).eval()
        self.device, self.np = device, np
        self.tf = create_val_transforms(size=224, is_crop=True)

    def preprocess(self, img):
        # dataset.py reads with cv2 and converts to RGB uint8, then the albumentations transform
        return self.tf(image=self.np.array(img.convert("RGB")))["image"]

    def forward(self, batch):
        # train.py:eval_model: score = 1 - softmax[:, 0] (= P(fake) with 2 classes), threshold 0.5
        return softmax_fake(self.model, batch, self.device, fake_index=1)


@register
class NPR(Detector):
    name = "npr"
    code = "https://github.com/chuangchuangtan/NPR-DeepfakeDetection"
    weights = []  # NPR.pth ships inside the repo
    checkpoint = "NPR.pth (repo root, ProGAN 4-class; the one used in README's test.py command)"
    preprocessing = "Resize((256, 256)) bilinear, no crop, ImageNet Normalize (test.py: no_resize=False, no_crop=True)"
    threshold = 0.5
    batch_size = 32

    def load(self, device):
        self.use_repo()
        from networks.resnet import resnet50
        self.model = resnet50(num_classes=1)
        # NPR.pth = {'model': state dict with 'module.' prefix}
        self.model.load_state_dict(state_dict(self.repo / "NPR.pth"), strict=True)
        self.model.to(device).eval()
        self.device = device

    def preprocess(self, img):
        # data/datasets.py:binary_dataset at test time with test.py's no_resize=False, no_crop=True (loadSize 256)
        return to_tensor(img, IMAGENET, size=(256, 256))

    def forward(self, batch):
        # validate.py: sigmoid of the single logit, fake if > 0.5
        return sigmoid_scores(self.model, batch, self.device)


@register
class FatFormer(Detector):
    name = "fatformer"
    code = "https://github.com/Michel-liu/FatFormer"
    weights = [
        ("gdrive", "1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj", "fatformer_4class.pth"),
        # CLIP backbone, README: save into `pretrained/` of the code base (models/clip/clip.py reads it relatively)
        ("url", "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
         "repo/pretrained/ViT-L-14.pt"),
    ]
    checkpoint = "released FatFormer checkpoint (CLIP:ViT-L/14, 4-class ProGAN, num_vit_adapter 3, num_context_embedding 8)"
    preprocessing = "Resize((256, 256)) + CenterCrop 224, ImageNet Normalize (utils/dataset.py test transform)"
    threshold = 0.5
    batch_size = 16

    def load(self, device):
        self.use_repo(chdir=True)  # clip.load opens "pretrained/ViT-L-14.pt" relative to cwd
        from types import SimpleNamespace
        from models import build_model
        # README inference command + main.py defaults
        args = SimpleNamespace(backbone="CLIP:ViT-L/14", num_classes=2, num_vit_adapter=3,
                               num_context_embedding=8, init_context_embedding="")
        self.model = build_model(args)
        # main.py: model.load_state_dict(checkpoint['model'])
        self.model.load_state_dict(state_dict(self.home / "fatformer_4class.pth"), strict=True)
        self.model.to(device).eval()
        self.device = device

    def preprocess(self, img):
        return to_tensor(img, IMAGENET, size=(256, 256), crop=224)

    def forward(self, batch):
        # main.py:evaluate: softmax[:, 1], fake if > 0.5
        return softmax_fake(self.model, batch, self.device, fake_index=1)
