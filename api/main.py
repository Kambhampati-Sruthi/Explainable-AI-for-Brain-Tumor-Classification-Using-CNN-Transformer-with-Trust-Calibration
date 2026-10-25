"""FastAPI backend. Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000   (from the project root)"""
import io, os, sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.deployment.inference import Engine

DEPLOY = Path(os.environ.get("DEPLOY_DIR", ROOT / "models" / "deploy"))
CKPT = os.environ.get("CLS_CKPT")                       # PyTorch checkpoint for /explain
MAX_BYTES = 10 * 1024 * 1024

@asynccontextmanager
async def lifespan(app):
    app.state.engine = Engine(DEPLOY, CKPT); yield

app = FastAPI(title="Brain Tumor XAI Decision Support (research prototype)", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"], allow_methods=["POST", "GET"], allow_headers=["*"])

def read_image(f: UploadFile) -> Image.Image:
    data = f.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES: raise HTTPException(413, "file larger than 10 MB")
    try: im = Image.open(io.BytesIO(data)); im.load()
    except Exception: raise HTTPException(400, "not a valid image file")
    if min(im.size) < 64: raise HTTPException(400, "image too small")
    return im

class Prediction(BaseModel):
    cls: str; confidence: float; calibration: str; probabilities: dict
#class AnalyzeResponse(BaseModel):
#    **{"class": str} if False else {}                    # (placeholder removed below)

from pydantic import Field

class AnalyzeResponse(BaseModel):
    class_: str = Field(alias="class")
    confidence: float
    calibration: str
    trust_level: str
    probabilities: dict
    tumor_area_px: int | None = None
    tumor_area_percent: float | None = None
    explanation: str
    heatmap: str                       # base64 PNG: Grad-CAM overlay (+ predicted tumor outline)
    disclaimer: str
    model_config = {"populate_by_name": True}

@app.get("/health")
@app.get("/api/health")
def health():
    engine = app.state.engine

    return {
        "status": "ok",
        "ready": True,
        "device": str(getattr(engine, "device", "CPU")),
        "max_mb": 10,
        "methods": ["Grad-CAM"],
        "integrity": {"status": "unknown"},
        "disclaimer": engine.cfg.get(
            "disclaimer",
            "Research prototype. Not for clinical use."
        ),
    }
@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...)):
    r = app.state.engine.predict(read_image(file))
    return Prediction(cls=r["cls"], confidence=r["conf"], calibration=r["calibration"], probabilities=r["probs"])

@app.post("/segment")
def segment(file: UploadFile = File(...)):
    e = app.state.engine; r = e.segment(read_image(file))
    return {"tumor_area_px": r["px"], "tumor_area_percent": r["pct"], "mask_overlay": e.overlay_png(r["arr"], mask=r["mask"]),
            "disclaimer": e.cfg["disclaimer"]}

@app.post("/explain")
def explain(file: UploadFile = File(...), layer: int = 5):
    e = app.state.engine
    if not e.torch_ckpt: raise HTTPException(503, "set CLS_CKPT to enable explanations (Grad-CAM needs the PyTorch model)")
    if layer not in (3, 5, 8): raise HTTPException(400, "layer must be 3, 5 or 8")
    im = read_image(file); p = e.predict(im); arr, cam = e.explain(im, k=p["k"], layer=layer)
    return {"class": p["cls"], "confidence": p["conf"], "layer": f"features[{layer}]", "heatmap": e.overlay_png(arr, heat=cam), "disclaimer": e.cfg["disclaimer"]}

@app.post("/analyze", response_model=AnalyzeResponse, response_model_by_alias=True)
def analyze(file: UploadFile = File(...)):
    """Classifier -> segmenter -> explanation -> trust level (rules from Stage 6; thresholds fixed before test evaluation)."""
    e = app.state.engine; im = read_image(file); p = e.predict(im); s = e.segment(im)
    tumor = p["cls"] != "no_tumor"; found = s["px"] > 50; agree = tumor == found
    cam, consist = None, None
    if e.torch_ckpt:
        arr, cam = e.explain(im, k=p["k"], layer=8)
        if tumor and s["px"] > 0: consist = float((cam * s["mask"]).sum() / max(cam.sum(), 1e-12))
    if p["conf"] < 0.80 or not agree: level = "LOW"
    elif p["conf"] < 0.95 or (consist is not None and consist < 0.30): level = "MEDIUM"
    else: level = "HIGH"
    if not tumor and not found: text = "No tumor predicted; the segmenter also found no tumor region."
    elif tumor and found:
        text = f"Predicted {p['cls']}; the segmenter outlines a region of {s['pct']:.1f}% of the slice."
        if consist is not None: text += f" {consist * 100:.0f}% of the heatmap's attention lies inside that region."
    else: text = "Classifier and segmenter DISAGREE; treat this result with caution and review the image manually."
    return AnalyzeResponse(**{"class": p["cls"]}, confidence=p["conf"], calibration=p["calibration"], trust_level=level, probabilities=p["probs"],
                           tumor_area_px=s["px"] if tumor else None, tumor_area_percent=s["pct"] if tumor else None, explanation=text,
                           heatmap=e.overlay_png(s["arr"], heat=cam, mask=s["mask"] if tumor else None), disclaimer=e.cfg["disclaimer"])

static_dir = ROOT / "webapp" / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(404, "Frontend UI index.html not found")