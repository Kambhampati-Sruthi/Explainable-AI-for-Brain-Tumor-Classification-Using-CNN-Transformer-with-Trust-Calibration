"""Everything the interactive UI needs from one request: model-view image, tumor outline/fill, three Grad-CAM
resolutions as transparent PNGs, and the trust level with human-readable reasons (same rule as Stage 6)."""
import base64, io, threading, time
import numpy as np
from PIL import Image
from scipy import ndimage

VERSION = "1.0.0"
LAYER_INFO = {"L8": dict(layer=8, grid=8, label="Coarse"), "L5": dict(layer=5, grid=16, label="Medium"), "L3": dict(layer=3, grid=32, label="Fine")}
LABEL = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumor", "no_tumor": "no tumor"}
_LOCK = threading.Lock()          # hooks on a shared torch model are not re-entrant: serialise the Grad-CAM part


def _png(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def heat_png(cam):
    """cam in [0,1] -> RGBA PNG, jet colours, transparent where attention is low (so it overlays cleanly)."""
    import matplotlib
    rgba = matplotlib.colormaps["jet"](np.clip(cam, 0, 1))
    rgba[..., 3] = np.clip((cam - 0.12) / 0.88, 0, 1) * 0.9
    return _png((rgba * 255).astype(np.uint8))

def outline_png(mask, thickness=2, color=(34, 255, 140)):
    edge = mask & ~ndimage.binary_erosion(mask, iterations=thickness)
    out = np.zeros(mask.shape + (4,), np.uint8); out[edge] = (*color, 255); return _png(out)

def fill_png(mask, color=(34, 255, 140), alpha=90):
    out = np.zeros(mask.shape + (4,), np.uint8); out[mask] = (*color, alpha); return _png(out)


def trust(conf, tumor, found, consist):
    """LOW: conf < 0.80 or classifier/segmenter disagree. MEDIUM: conf < 0.95 or attention mostly outside the predicted region.
    HIGH otherwise. (Thresholds fixed in Stage 6 before the test evaluation.) 'HIGH' does not mean 'correct'."""
    agree = tumor == found; reasons = []
    if conf < 0.80: reasons.append("Model confidence is below 80%.")
    elif conf < 0.95: reasons.append("Model confidence is below 95%.")
    if not agree: reasons.append("The classifier and the segmenter disagree about whether a tumor is present.")
    if consist is not None and consist < 0.30:
        reasons.append(f"Only {consist * 100:.0f}% of the heatmap's attention lies inside the predicted tumor region.")
    if conf < 0.80 or not agree: level = "LOW"
    elif conf < 0.95 or (consist is not None and consist < 0.30): level = "MEDIUM"
    else: level = "HIGH"
    if level == "HIGH":
        reasons.append("Confidence is high, the segmenter agrees" + (", and the attention overlaps the predicted region." if consist is not None else "."))
    return level, reasons, agree


def _ensure_torch(engine):
    if engine._torch is None:
        import torch
        from src.classification.model import build_efficientnet_b0
        m = build_efficientnet_b0(pretrained=False)
        m.load_state_dict(torch.load(engine.torch_ckpt, map_location="cpu")["model"]); engine._torch = m.eval()
    return engine._torch


def analyze_layers(engine, im, default_layer="L5"):
    t0 = time.perf_counter()
    p, s = engine.predict(im), engine.segment(im)
    arr, xc, _ = engine.prep(im)
    tumor, found = p["cls"] != "no_tumor", s["px"] > 50
    layers, cams, consist = {}, {}, {}
    if engine.torch_ckpt:
        import torch
        from src.xai.methods import MultiGradCAM
        with _LOCK:
            m = _ensure_torch(engine)
            mc = MultiGradCAM(m, {k: m.features[v["layer"]] for k, v in LAYER_INFO.items()})
            try: out, _, _ = mc(torch.from_numpy(xc), target=p["k"])
            finally: mc.remove()
        for k, c in out.items():
            c = c[0].numpy(); cams[k] = c; layers[k] = heat_png(c)
            if tumor and s["px"] > 0: consist[k] = float((c * s["mask"]).sum() / max(c.sum(), 1e-12))
    c8 = consist.get("L8")
    level, reasons, agree = trust(p["conf"], tumor, found, c8)
    ys, xs = np.where(s["mask"]); bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    if not tumor and not found: text = "No tumor class was predicted, and the segmenter found no tumor region."
    elif tumor and found:
        text = f"Predicted {LABEL[p['cls']]}. The segmenter outlines a region covering {s['pct']:.1f}% of the model view."
        if c8 is not None: text += f" About {c8 * 100:.0f}% of the heatmap's attention (coarse layer) lies inside that region."
    elif tumor: text = "A tumor class was predicted, but the segmenter found no clear region. The two models disagree: review the image manually."
    else: text = "No tumor class was predicted, but the segmenter found a candidate region. The two models disagree: review the image manually."
    has_mask = s["px"] > 0
    payload = {
        "version": VERSION, "timings_ms": {"total": round((time.perf_counter() - t0) * 1000)},
        "preprocessing": {"original_size": list(im.size), "model_view": "grayscale, padded to a square, resized to 256 x 256"},
        "prediction": {"class": p["cls"], "confidence": p["conf"], "probabilities": p["probs"], "calibration": p["calibration"]},
        "segmentation": {"found": found, "has_mask": has_mask, "area_px": s["px"], "area_percent": s["pct"], "bbox": bbox},
        "explanation": {"layers": layers, "default_layer": default_layer if default_layer in layers else (next(iter(layers), None)),
                        "consistency": c8, "text": text, "layer_info": LAYER_INFO},
        "trust": {"level": level, "reasons": reasons, "signals": {"confidence": p["conf"], "agreement": agree, "consistency": c8}},
        "images": {"base": _png(arr), "outline": outline_png(s["mask"]) if has_mask else None, "fill": fill_png(s["mask"]) if has_mask else None},
        "disclaimer": engine.cfg["disclaimer"]}
    return payload, {"arr": arr, "cams": cams, "mask": s["mask"]}