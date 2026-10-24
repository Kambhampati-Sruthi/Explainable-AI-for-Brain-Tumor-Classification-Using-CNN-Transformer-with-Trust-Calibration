"""FastAPI server for the local web app.   python webapp/run.py   ->  http://127.0.0.1:8000
Images are processed in memory only: nothing is written to disk and responses are marked no-store."""
import logging, threading
from contextlib import asynccontextmanager
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from webapp import service

log = logging.getLogger("btx"); STATIC = Path(__file__).parent / "static"


def default_engine():
    from local_app.pipeline import BrainTumorXAI               # imports torch lazily, so the server can still start and explain what is missing
    return BrainTumorXAI()


def create_app(engine_factory=None):
    lock = threading.Lock()                                    # Grad-CAM hooks are not thread-safe: one analysis at a time

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine, app.state.error = None, None
        try: app.state.engine = (engine_factory or default_engine)()
        except Exception as e: app.state.error = f"{type(e).__name__}: {e}"; log.error("model loading failed: %s", app.state.error)
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
        except Exception:
            log.exception("analysis failed"); raise HTTPException(500, "The analysis failed on the server. See the server console for details.")

    @app.post("/api/compare")
    def compare(file: UploadFile = File(...), layer: int = Form(8)):
        e = engine_or_503(); raw = read(file)
        try:
            with lock: return service.compare_request(e, raw, layer)
        except service.InputError as err: raise HTTPException(err.status, str(err))
        except Exception:
            log.exception("comparison failed"); raise HTTPException(500, "The comparison failed on the server. See the server console for details.")

    @app.get("/")
    def index(): return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()