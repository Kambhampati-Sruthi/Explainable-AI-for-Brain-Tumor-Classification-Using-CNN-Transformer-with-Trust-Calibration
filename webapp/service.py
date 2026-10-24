"""Framework-independent request handling: validate the upload, run the engine, build the JSON response.
(No FastAPI / torch imports here, so it can be unit-tested with a stub engine.)"""
import hashlib, io, time, uuid, warnings
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from local_app import core
from webapp import render

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 15 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF", "WEBP"}
LAYERS, BACKENDS = (3, 5, 8), ("lib", "own")
DEFAULT_METHODS = ["Grad-CAM", "Grad-CAM++", "HiRes-CAM", "XGrad-CAM", "Layer-CAM", "Eigen-CAM", "Score-CAM (slow)"]
Image.MAX_IMAGE_PIXELS = 40_000_000


class InputError(Exception):
    def __init__(self, msg, status=400): super().__init__(msg); self.status = status


# ------------------------------------------------------------------ input validation
def load_image(raw: bytes):
    """bytes -> (PIL 'L' image, info dict, warnings list). Raises InputError with a user-readable message."""
    if not raw: raise InputError("The upload is empty.")
    if len(raw) > MAX_BYTES: raise InputError(f"The file is larger than {MAX_BYTES // 2**20} MB.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(raw)); fmt = im.format; im.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning): raise InputError("The image has too many pixels.", 413)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise InputError("This file could not be read as an image. Use PNG, JPEG, BMP, TIFF or WebP (DICOM is not supported: export the slice as PNG).", 415)
    if fmt not in ALLOWED_FORMATS: raise InputError(f"Unsupported image format {fmt}. Use PNG, JPEG, BMP, TIFF or WebP.", 415)
    if getattr(im, "n_frames", 1) > 1: im.seek(0)                                            # multi-frame TIFF/WebP: first frame only
    w, h = im.size
    if min(w, h) < 64: raise InputError("The image is too small (at least 64 px on the short side).", 422)
    warn, depth16 = [], im.mode in ("I;16", "I;16L", "I;16B", "I", "F")
    if depth16:
        a = np.asarray(im).astype(np.float32); lo, hi = np.percentile(a, [0.5, 99.5])
        gray = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8); color = 0.0
        warn.append("16-bit image: rescaled to 8-bit using its 0.5-99.5 percentile range.")
    else:
        rgb = np.asarray(im.convert("RGB")).astype(np.int16); color = float((rgb.max(2) - rgb.min(2)).mean()); gray = np.asarray(im.convert("L"))
    if float(gray.std()) < 2: raise InputError("The image is almost uniform, so there is no visible anatomy to analyse.", 422)
    if color > 12: warn.append("This looks like a colour image or photo. The models were trained on grayscale MRI slices, so results may be unreliable.")
    if min(w, h) < 128: warn.append(f"Low resolution ({w}x{h}); the slice is upsampled to 256x256.")
    if max(w, h) / min(w, h) > 1.35: warn.append("Non-square image: the classifier input is stretched and the segmenter input padded to a square, exactly as during training.")
    info = dict(width=w, height=h, format=fmt, mode=im.mode, bytes=len(raw), mean_color_spread=round(color, 1))
    return Image.fromarray(gray), info, warn


def _check_params(engine, method, layer, backend):
    names = getattr(engine, "method_names", DEFAULT_METHODS)
    if method not in names: raise InputError(f"Unknown method '{method}'.", 422)
    if layer not in LAYERS: raise InputError("Layer must be 3, 5 or 8.", 422)
    if backend not in BACKENDS: raise InputError("Backend must be 'lib' or 'own'.", 422)
    if backend == "own" and method != "Grad-CAM": raise InputError("The project's own implementation only provides Grad-CAM.", 422)


# ------------------------------------------------------------------ response building
def trust_checks(r):
    c, agree, cons = r["conf"], r["tumor"] == r["found"], r["consistency"]
    out = [dict(id="confidence", label="Calibrated confidence", value=f"{c * 100:.1f}%", rule=f"at least {core.CONF_HIGH * 100:.0f}% for HIGH, at least {core.CONF_LOW * 100:.0f}% for MEDIUM",
                status="pass" if c >= core.CONF_HIGH else ("warn" if c >= core.CONF_LOW else "fail")),
           dict(id="agreement", label="Classifier and segmenter agree", value="yes" if agree else "no",
                rule=f"tumor class if and only if a region of more than {core.MIN_TUMOR_PX} px is found", status="pass" if agree else "fail")]
    if cons is None: out.append(dict(id="attention", label="Heat-map inside the tumor region", value="n/a", rule="only checked for tumor predictions with a region", status="na"))
    else: out.append(dict(id="attention", label="Heat-map inside the tumor region", value=f"{cons * 100:.0f}%", rule=f"at least {core.CONSIST_MIN * 100:.0f}% of Grad-CAM attention inside the region",
                          status="pass" if cons >= core.CONSIST_MIN else "warn"))
    return out


TRUST_TEXT = {"HIGH": "All trust checks passed. Still a research prototype: a clinician must review every case.",
              "MEDIUM": "At least one check is borderline. Review the heat-map and the outline carefully before relying on this result.",
              "LOW": "Low confidence, or the classifier and segmenter disagree. Do not rely on this result."}


def panel_b64(r):
    fig = core.render_panel(r); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130)
    import base64; return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_response(r, info, warn, raw, ms):
    cfg_T = r["T"]; has_region = r["px"] > 0
    fill, edge = render.mask_layers(r["mask"]) if has_region else (None, None)
    seg = dict(region_found=bool(r["found"]), shown_in_report=bool(r["tumor"]), area_px=int(r["px"]), area_percent=round(float(r["pct"]), 2),
               disagrees_with_classifier=bool(r["tumor"] != r["found"]), **render.measure(r["mask"]) if has_region else dict(present=False))
    ws = list(warn)
    if r["tumor"] and r["found"] and r["pct"] < 0.3: ws.append("The segmented region is very small; small regions are the hardest case for both models.")
    if r["tumor"] != r["found"]: ws.append("The classifier and the segmenter disagree about whether a tumor is present.")
    return dict(
        ok=True,
        prediction=dict(**{"class": r["cls"]}, confidence=round(r["conf"], 4), calibration_band=r["calibration"], temperature=round(cfg_T, 4),
                        probabilities={k: round(v, 4) for k, v in r["probs"].items()}, probabilities_uncalibrated={k: round(v, 4) for k, v in r["probs_raw"].items()}),
        trust=dict(level=r["trust"], summary=TRUST_TEXT[r["trust"]], checks=trust_checks(r)),
        segmentation=seg,
        explanation=dict(method=r["method"], layer=r["layer"], backend=r["backend"], label=r["method_label"], target_class=r["cls"],
                         cam_inside_mask=None if r["consistency"] is None else round(r["consistency"], 4),
                         library_vs_project_gradcam_corr=None if r["lib_vs_own_corr"] is None else round(r["lib_vs_own_corr"], 4),
                         note="The heat-map shows where the classifier looked for its predicted class. The trust check always uses the project's validated Grad-CAM on features[8]."),
        text=r["text"], warnings=ws,
        images=dict(size=render.OUT, base=render.b64_png(render.base_layer(r["arr_c"])), heat=render.b64_png(render.heat_rgba(r["cam"])),
                    mask_fill=render.b64_png(fill) if fill is not None else None, mask_edge=render.b64_png(edge) if edge is not None else None, panel=panel_b64(r)),
        meta=dict(request_id=uuid.uuid4().hex[:8], input_sha256=hashlib.sha256(raw).hexdigest()[:16], input=info, runtime_ms=int(ms), disclaimer=r["disclaimer"]))


def analyze_request(engine, raw, method="Grad-CAM", layer=8, backend="lib"):
    _check_params(engine, method, layer, backend); im, info, warn = load_image(raw); t0 = time.perf_counter()
    r = engine.analyze(im, method, layer, backend)
    return build_response(r, info, warn, raw, (time.perf_counter() - t0) * 1000)


def compare_request(engine, raw, layer=8):
    if layer not in LAYERS: raise InputError("Layer must be 3, 5 or 8.", 422)
    im, info, warn = load_image(raw); t0 = time.perf_counter(); c = engine.compare(im, layer)
    item = lambda d: dict(label=d["label"], layer=d["layer"], error=d["error"],
                          image=None if d["cam"] is None else render.b64_png(render.overlay_rgb(c["arr_c"], d["cam"], c["mask"] if c["tumor"] else None)))
    return dict(ok=True, target_class=c["cls"], layer=layer, methods=[item(d) for d in c["methods"]], layers=[item(d) for d in c["layers"]],
                meta=dict(input_sha256=hashlib.sha256(raw).hexdigest()[:16], runtime_ms=int((time.perf_counter() - t0) * 1000)))


# ------------------------------------------------------------------ model-file integrity
def check_integrity(deploy_dir=None):
    d = Path(deploy_dir or ROOT / "models" / "deploy"); sums = d / "SHA256SUMS.txt"; files = {}
    if not sums.exists(): return dict(status="unknown", files={}, note="SHA256SUMS.txt not found; model files could not be verified.")
    for line in sums.read_text().splitlines():
        if not line.strip(): continue
        digest, name = line.split(maxsplit=1); name = name.strip().lstrip("*"); f = d / name
        if not name.endswith((".onnx", ".pt")): continue          # configs (e.g. deploy_config.json) are edited on purpose; only model weights are integrity-checked
        if not f.exists(): files[name] = "absent"; continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
        files[name] = "ok" if h.hexdigest() == digest else "MISMATCH"
    present = [v for v in files.values() if v != "absent"]
    status = "mismatch" if "MISMATCH" in present else ("verified" if present else "unknown")
    return dict(status=status, files=files, note="The segmentation checkpoint in models/ has no recorded checksum and is not covered.")