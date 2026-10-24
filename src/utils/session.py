"""Restore a notebook's state from saved files; load models by checkpoint path."""
import json, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, torch

ROOT = Path("/kaggle/working/BrainTumor_XAI_Project")
CLASS_ORDER = ["glioma", "meningioma", "no_tumor", "pituitary"]


def bootstrap(gpu=True):
    from src.utils.seed import set_seed
    set_seed(42)
    S = SimpleNamespace(ROOT=ROOT, CLASS_ORDER=CLASS_ORDER)
    S.TAB, S.FIG, S.MOD, S.CFG, S.CACHE = (ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "models",
                                           ROOT / "configs", ROOT / "data/cache")
    for d in (S.TAB, S.FIG, S.MOD, S.CFG, ROOT / "docs", ROOT / "api", ROOT / "tests"):
        d.mkdir(parents=True, exist_ok=True)
    S.pre = json.load(open(S.CFG / "preprocessing.json"))
    S.STATS = (S.pre["train_mean"], S.pre["train_std"])
    S.cache_index = pd.read_csv(S.CACHE / "cache_index_pad_256.csv")
    S.DEVICE = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    S.X, S.M = np.load(S.CACHE / "images_pad_256.npy"), np.load(S.CACHE / "masks_pad_256.npy")
    fx = hashlib.sha256(S.X.tobytes()).hexdigest()[:16]
    assert fx == "fc5ecbc245817db5", f"cache fingerprint differs from the verified one: {fx}"
    p = S.CFG / "classification_final.json"
    S.cls_fin = json.load(open(p)) if p.exists() else None
    if S.cls_fin and S.cls_fin["arrays"] == "stretch":
        S.Xc, S.Mc = np.load(S.CACHE / "images_stretch_256.npy"), np.load(S.CACHE / "masks_stretch_256.npy")
    else:
        S.Xc, S.Mc = S.X, S.M                                   # arrays the CLASSIFIER was trained on
    return S


def load_cls_path(path, S):
    from src.classification.model import build_efficientnet_b0
    ck = torch.load(path, map_location=S.DEVICE)
    m = build_efficientnet_b0(pretrained=False).to(S.DEVICE)
    m.load_state_dict(ck["model"])
    return m.eval(), ck


def primary_cls(S, seeds=(0, 1, 2)):
    """Seed of the final classifier with the best VALIDATION macro-F1 (never the test score)."""
    return max(seeds, key=lambda s: torch.load(S.MOD / f"final_s{s}_best.pt", map_location="cpu")["val_macro_f1"])


def load_seg_path(path, S):
    from src.segmentation.model import UNetPlusPlus, EffB1Encoder
    ck = torch.load(path, map_location=S.DEVICE)
    cfg = ck.get("cfg", {})
    m = UNetPlusPlus(pretrained=False, nested=cfg.get("nested", True))
    if cfg.get("variant", "b0") == "b1":
        m.encoder = EffB1Encoder(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(S.DEVICE).eval(), ck
