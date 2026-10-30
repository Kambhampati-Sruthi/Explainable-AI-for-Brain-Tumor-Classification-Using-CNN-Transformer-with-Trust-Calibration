"""FastAPI server for the local web app.   uvicorn api.main:app --port 8000   ->  http://127.0.0.1:8000
Images are processed in memory only: nothing is written to disk and responses are marked no-store."""
import importlib, logging, os, threading
from contextlib import asynccontextmanager
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from api import service

try: from api import metrics                        # OPTIONAL: the "Model performance" page. Without api/metrics.py the server still runs.
except ImportError: metrics = None

log = logging.getLogger("btx"); STATIC = Path(__file__).parent / "static"

# Packages / project modules that are imported LAZILY, i.e. only when the first analysis runs. If one is missing on the server the models
# still load (the page says "ready") and then every upload fails with "analysis failed". Checking them at start-up shows the cause at once.
RUNTIME_IMPORTS = ["cv2", "PIL", "scipy.ndimage", "tqdm", "matplotlib.backends.backend_agg", "pytorch_grad_cam", "torchvision",
                   "src.preprocessing.cache", "src.xai.methods"]


def runtime_problems():
    bad = []
    for name in RUNTIME_IMPORTS:
        try: importlib.import_module(name)
        except Exception as e: bad.append(f"{name} ({type(e).__name__}: {e})")
    return bad


def default_engine():
    from api.pipeline import BrainTumorXAI               # imports torch lazily, so the server can still start and explain what is missing
    return BrainTumorXAI()


def create_app(engine_factory=None):
    lock = threading.Lock()                                    # Grad-CAM hooks are not thread-safe: one analysis at a time

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine, app.state.error = None, None
        try: app.state.engine = (engine_factory or default_engine)()
        except Exception as e: app.state.error = f"{type(e).__name__}: {e}"; log.error("model loading failed: %s", app.state.error)
        if app.state.engine is not None and engine_factory is None:                       # real models: also verify the lazily imported packages
            bad = runtime_problems()
            if bad:
                app.state.error = "Missing or broken packages/files: " + "; ".join(bad) + ". Add them to requirements.txt (or commit the files) and redeploy."
                log.error(app.state.error); app.state.engine = None
        app.state.integrity = service.check_integrity()
        yield

    app = FastAPI(title="Brain tumor XAI", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def no_store(request, call_next):
        resp = await call_next(request); resp.headers["Cache-Control"] = "no-store"; resp.headers["X-Content-Type-Options"] = "nosniff"; return resp

    def engine_or_503():
        if app.state.engine is None: raise HTTPException(503, f"Models are not loaded: {app.state.error}")
        return app.state.engine

    def read(upload: UploadFile):
        return upload.file.read(service.MAX_BYTES + 1)

    def debug_note(err):                                       # set BTX_DEBUG=1 on the server to see the exception in the browser too
        return f" ({type(err).__name__}: {err})" if os.environ.get("BTX_DEBUG") else ""

    @app.get("/api/health")
    def health():
        e = app.state.engine
        return dict(ready=e is not None, error=app.state.error, integrity=app.state.integrity, device=str(getattr(e, "device", "n/a")),
                    methods=getattr(e, "method_names", service.DEFAULT_METHODS), layers=list(service.LAYERS), max_mb=service.MAX_BYTES // 2**20,
                    classes=(e.cfg["classes"] if e is not None else []), disclaimer=(e.cfg["disclaimer"] if e is not None else "Research prototype. Not a medical device."))

    @app.post("/api/analyze")
    def analyze(file: UploadFile = File(...), method: str = Form("Grad-CAM"), layer: int = Form(8), backend: str = Form("lib")):
        e = engine_or_503(); raw = read(file)
        try:
            with lock: return service.analyze_request(e, raw, method, layer, backend)
        except service.InputError as err: raise HTTPException(err.status, str(err))
        except Exception as err:
            log.exception("analysis failed"); raise HTTPException(500, "The analysis failed on the server. See the server console for details." + debug_note(err))

    @app.post("/api/compare")
    def compare(file: UploadFile = File(...), layer: int = Form(8)):
        e = engine_or_503(); raw = read(file)
        try:
            with lock: return service.compare_request(e, raw, layer)
        except service.InputError as err: raise HTTPException(err.status, str(err))
        except Exception as err:
            log.exception("comparison failed"); raise HTTPException(500, "The comparison failed on the server. See the server console for details." + debug_note(err))

    @app.get("/")
    def index(): return FileResponse(STATIC / "index.html")

    if metrics is not None and (STATIC / "metrics.html").exists():                    # optional "Model performance" page
        @app.get("/metrics")
        def metrics_page(): return FileResponse(STATIC / "metrics.html")

        @app.get("/api/metrics")
        def get_metrics():
            try: return metrics.load()
            except Exception as err:
                log.exception("metrics failed"); raise HTTPException(500, "The metrics could not be computed. See the server console for details." + debug_note(err))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():                                             # serves api/static/favicon.ico when it exists, else 204 "no content" (no 404)
        f = STATIC / "favicon.ico"
        return FileResponse(f, media_type="image/x-icon") if f.exists() else Response(status_code=204)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()


if __name__ == "__main__":                                       # python api/main.py
    import uvicorn; uvicorn.run("api.main:app", host="127.0.0.1", port=8000)   # 127.0.0.1 = this computer only (medical images: do not expose publicly)