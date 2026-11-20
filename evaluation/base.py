"""What every detector implements, and the automatic download of code and weights.

A detector is one class in detectors.py with:
  code      official git repository, cloned into <cache>/<name>/repo (its code is imported from there)
  weights   list of downloads into <cache>/<name>/ (see `download`)
  load()    build the model as in the official test script and load the released weights
  preprocess(img) / forward(batch)   official test-time preprocessing and a fake score per image
The cache is $CSS_CACHE (default ~/.cache/css-eval). `fetch()` runs before `load()` and skips
whatever is already there, so the first run downloads and the next ones do not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REGISTRY: dict[str, type["Detector"]] = {}
CACHE = Path(os.environ.get("CSS_CACHE", "~/.cache/css-eval")).expanduser()


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


class Detector:
    name: str = ""
    code: str = ""            # official repository (git URL), "" if the model needs no repo code
    commit: str = ""          # optional: pin the repo to this commit
    weights: list = []        # downloads, see `download`
    checkpoint: str = ""      # which released checkpoint is used (no training, no fine-tuning)
    preprocessing: str = ""   # official test-time preprocessing, as a note
    threshold: float = 0.5    # released decision threshold on `score`
    batch_size: int = 1       # >1 only if preprocess() returns fixed-size tensors
    seed: int = 0             # for detectors with random test-time steps (set by detect.py --seed)

    @property
    def home(self) -> Path:
        return CACHE / self.name

    @property
    def repo(self) -> Path:
        return self.home / "repo"

    def fetch(self) -> None:
        """Clone the official repo and download the released weights (only what is missing)."""
        self.home.mkdir(parents=True, exist_ok=True)
        if self.code and not self.repo.exists():
            run(["git", "clone", "--depth", "50", self.code, str(self.repo)])
            if self.commit:
                run(["git", "-C", str(self.repo), "checkout", self.commit])
        for spec in self.weights:
            download(spec, self)

    def use_repo(self, chdir: bool = False) -> None:
        """Make the official code importable (and optionally run from its folder, for relative paths)."""
        sys.path.insert(0, str(self.repo))
        if chdir:
            os.chdir(self.repo)

    def load(self, device: str) -> None:
        raise NotImplementedError(f"{self.name}: adapter not written")

    def preprocess(self, img):
        """PIL RGB image -> model input for one image (official preprocessing)."""
        return img

    def forward(self, batch: list) -> list[float]:
        """List of preprocessed images -> one score per image, higher = more fake (e.g. P(fake))."""
        raise NotImplementedError


# ------------------------------------------------------------------------------ downloads
def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def download(spec: tuple, det: Detector) -> None:
    """One download. `dest` is relative to the detector's cache folder; a .zip is extracted there.

      ("url", url, dest)                       plain HTTP(S) file
      ("gdrive", file_id, dest)                Google Drive file          (pip install gdown)
      ("gdrive_folder", folder_id, dest)       Google Drive folder        (pip install gdown)
      ("hf", repo_id, dest)                    Hugging Face snapshot      (pip install huggingface_hub)
      ("hf_file", repo_id, filename, dest)     one file of a Hugging Face repo
      ("modelscope", repo_id, filename, dest)  one file of a ModelScope model repo
      ("modelscope_dataset", repo_id, filename, dest)  one file of a ModelScope dataset repo
      ("lfs", )                                git lfs pull inside the cloned repo
    """
    kind, *args = spec
    if kind == "lfs":
        run(["git", "-C", str(det.repo), "lfs", "pull"])
        return
    dest = det.home / args[-1]
    if dest.exists() or (dest.suffix == ".zip" and dest.with_suffix("").exists()):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if kind == "url":
        urllib.request.urlretrieve(args[0], tmp)
        tmp.rename(dest)
    elif kind == "modelscope":
        repo_id, filename = args[0], args[1]
        urllib.request.urlretrieve(
            f"https://www.modelscope.cn/models/{repo_id}/resolve/master/{filename}", tmp)
        tmp.rename(dest)
    elif kind == "modelscope_dataset":
        repo_id, filename = args[0], args[1]
        urllib.request.urlretrieve(f"https://www.modelscope.cn/api/v1/datasets/{repo_id}/repo"
                                   f"?Revision=master&FilePath={filename}", tmp)
        tmp.rename(dest)
    elif kind == "gdrive":
        import gdown
        gdown.download(id=args[0], output=str(tmp), quiet=False)
        tmp.rename(dest)
    elif kind == "gdrive_folder":
        import gdown
        gdown.download_folder(id=args[0], output=str(tmp), quiet=False)
        tmp.rename(dest)
    elif kind == "hf":
        from huggingface_hub import snapshot_download
        snapshot_download(args[0], local_dir=str(tmp))
        tmp.rename(dest)
    elif kind == "hf_file":
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(args[0], args[1])
        shutil.copy(p, dest)
    else:
        raise ValueError(f"unknown download kind {kind!r}")
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.with_suffix(""))
        dest.unlink()
        dest.with_suffix("").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------- torch helpers
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def find(root: Path, pattern: str) -> Path:
    """The single file matching `pattern` under `root` (zips and Drive folders nest unpredictably)."""
    hits = sorted(Path(root).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return hits[0]


def state_dict(ckpt, keys=("model", "state_dict", "net"), strip=("module.",)) -> dict:
    """A released checkpoint as a plain state dict: unwraps the usual containers, strips prefixes."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in keys:
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
    for p in strip:
        sd = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}
    return sd


def to_tensor(img, mean_std, size=None, crop=None):
    """ToTensor + Normalize, with an optional resize and center crop."""
    import torchvision.transforms as T
    ops = ([T.Resize(size)] if size else []) + ([T.CenterCrop(crop)] if crop else [])
    return T.Compose(ops + [T.ToTensor(), T.Normalize(*mean_std)])(img)


def sigmoid_scores(model, batch, device):
    """Stack, run, sigmoid of a 1-logit output."""
    import torch
    with torch.no_grad():
        return model(torch.stack(batch).to(device)).sigmoid().flatten().tolist()


def softmax_fake(model, batch, device, fake_index=1):
    """Stack, run, softmax probability of the fake class of a 2-logit output."""
    import torch
    with torch.no_grad():
        return model(torch.stack(batch).to(device)).softmax(-1)[:, fake_index].tolist()
