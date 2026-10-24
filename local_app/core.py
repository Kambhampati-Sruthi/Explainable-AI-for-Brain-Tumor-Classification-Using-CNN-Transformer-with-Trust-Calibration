"""Torch-free helpers for the local app: pre-processing, frame alignment, calibration band, trust level, rendering.
Everything here mirrors what the Kaggle notebook / api did, so numbers stay comparable."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

MEAN_IN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CONF_LOW, CONF_HIGH, CONSIST_MIN, MIN_TUMOR_PX = 0.80, 0.95, 0.30, 50     # trust rules fixed in notebook Stage 6 (before test evaluation)


def load_cfg(deploy_dir):
    return json.load(open(Path(deploy_dir) / "deploy_config.json"))        # (json accepts the NaN values written by the notebook)


def _norm(arr_u8, mean, std):
    x = (arr_u8.astype(np.float32) / 255.0)[None, None].repeat(3, 1)
    return ((x - np.array(mean, np.float32).reshape(1, 3, 1, 1)) / np.array(std, np.float32).reshape(1, 3, 1, 1)).astype(np.float32)


def prepare(im: Image.Image, cfg):
    """Any PIL image -> classifier input (its own resize policy: 'stretch') and segmenter input ('pad'), exactly as in training."""
    from src.preprocessing.cache import process_image                      # the SAME function used to build the training cache
    gray = np.array(im.convert("L")); size = cfg["img_size"]
    cp = cfg.get("cls_resize_policy", cfg["resize_policy"]); sp = cfg.get("seg_resize_policy", cfg["resize_policy"])
    arr_c, arr_s = process_image(gray, cp, size), process_image(gray, sp, size)
    if cfg["cls_norm"] == "imagenet": xc = _norm(arr_c, *MEAN_IN)
    elif cfg["cls_norm"] == "dataset": xc = _norm(arr_c, (cfg["stats"][0],) * 3, (cfg["stats"][1],) * 3)
    else: xc = _norm(arr_c, (0, 0, 0), (1, 1, 1))
    return dict(arr_c=arr_c, arr_s=arr_s, xc=xc, xs=_norm(arr_s, *MEAN_IN), hw=gray.shape)


def to_cls_frame(mask, hw, cfg):
    """Re-express the segmenter's mask (pad frame) in the classifier's frame (stretch) so mask, heat-map and overlay line up."""
    cp = cfg.get("cls_resize_policy", cfg["resize_policy"]); sp = cfg.get("seg_resize_policy", cfg["resize_policy"])
    if cp == sp or not (sp == "pad" and cp == "stretch"): return mask
    h, w = hw; size = cfg["img_size"]; side = max(h, w)
    hs, ws = max(1, round(h * size / side)), max(1, round(w * size / side)); top, left = (size - hs) // 2, (size - ws) // 2
    crop = mask[top:top + hs, left:left + ws].astype(np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def band(conf, cfg):
    for b in cfg["bands"]:
        if b["lo"] <= conf < b["hi"] or (conf >= 1.0 and b["hi"] >= 1.0): return b["label"]
    return "Insufficient data"


def trust_level(conf, agree, consist):
    if conf < CONF_LOW or not agree: return "LOW"
    if conf < CONF_HIGH or (consist is not None and consist < CONSIST_MIN): return "MEDIUM"
    return "HIGH"


def explanation_text(cls, tumor, found, pct, consist):
    if not tumor and not found: return "No tumor predicted; the segmenter also found no tumor region."
    if tumor and found:
        t = f"Predicted {cls}; the segmenter outlines a region of {pct:.1f}% of the slice."
        return t + (f" {consist * 100:.0f}% of the Grad-CAM attention lies inside that region." if consist is not None else "")
    return "Classifier and segmenter DISAGREE; treat this result with caution and review the image manually."


def overlay(arr_u8, cam=None, mask=None, alpha=0.45):
    """uint8 gray (H,W) [+ heat-map in 0..1] [+ bool mask outline in green] -> uint8 RGB."""
    import matplotlib
    rgb = np.repeat(arr_u8[..., None], 3, 2).astype(np.float32) / 255
    if cam is not None: rgb = (1 - alpha) * rgb + alpha * matplotlib.colormaps["jet"](np.clip(cam, 0, 1))[..., :3]
    if mask is not None and mask.any():
        from scipy import ndimage
        rgb[mask & ~ndimage.binary_erosion(mask)] = [0, 1, 0]
    return (rgb * 255).astype(np.uint8)


def render_panel(r, path=None):
    """4-panel figure: input | segmentation | explanation | calibrated probabilities. Returns the Figure (and saves it if path is given)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(16, 5.2)); FigureCanvasAgg(fig); ax = fig.subplots(1, 4)
    ax[0].imshow(r["arr_c"], cmap="gray"); ax[0].set_title("Input slice")
    ax[1].imshow(overlay(r["arr_c"], mask=r["mask"] if r["tumor"] else None))
    ax[1].set_title(f"Segmentation ({r['pct']:.1f}% of slice)" if r["tumor"] else "Segmentation (not shown: no tumor predicted)", fontsize=10)
    ax[2].imshow(overlay(r["arr_c"], cam=r["cam"], mask=r["mask"] if r["tumor"] else None)); ax[2].set_title(f"{r['method_label']} for '{r['cls']}'", fontsize=10)
    for a in ax[:3]: a.axis("off")
    names = list(r["probs"]); vals = [r["probs"][n] for n in names]
    ax[3].barh(names, vals, color=["#3b6fd4" if n == r["cls"] else "#9aa0a8" for n in names]); ax[3].set_xlim(0, 1); ax[3].invert_yaxis()
    ax[3].set_title(f"Calibrated probabilities (T={r['T']:.2f})", fontsize=10)
    for i, v in enumerate(vals): ax[3].text(min(v + 0.02, 0.8), i, f"{v * 100:.1f}%", va="center")
    col = {"HIGH": "#1b8a3a", "MEDIUM": "#b7791f", "LOW": "#c0392b"}[r["trust"]]
    fig.suptitle(f"{r['cls']}  |  confidence {r['conf'] * 100:.1f}% ({r['calibration']})  |  TRUST: {r['trust']}", color=col, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, r["text"] + "   " + r["disclaimer"], ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    if path: fig.savefig(path, dpi=140)
    return fig