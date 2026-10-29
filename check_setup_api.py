"""Run from the project root:  python check_setup_api.py
Checks the api/ layout (single FastAPI backend + static page), your files, packages, and loads the models."""
import importlib, json, re, sys, traceback
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
ok = True


def need(path, note="", fatal=True):
    global ok
    good = (R / path).exists()
    print(("OK       " if good else "MISSING  ") + path + ("" if good or not note else "    <- " + note))
    if fatal: ok &= good
    return good


print("Project folder:", R, "\n")
print("--- folders and files")
for p in ("src", "configs", "local_app/core.py", "local_app/pipeline.py", "api/main.py", "api/service.py", "api/render.py",
          "api/static/index.html", "api/static/app.js", "api/static/app.css"):
    need(p)
need("api/__init__.py", "optional", fatal=False)
need("app.py", "needed only for Hugging Face; locally use: python -m uvicorn api.main:app", fatal=False)
need("models/deploy/classifier_best.pt", "copy it from the btx_models download")
need("models/deploy/deploy_config.json")
need("models/deploy/SHA256SUMS.txt", "optional (integrity check)", fatal=False)
if need("configs/segmentation_primary.json"):
    prim = json.load(open(R / "configs/segmentation_primary.json")).get("primary")
    print(f"          segmenter named in that file: {prim}")
    need(f"models/{prim}_best.pt", "segmenter checkpoint, must be directly inside models/")

print("\n--- api/ must not depend on webapp/ (so webapp/ can be removed later)")
for f in ("api/main.py", "api/service.py", "api/render.py"):
    if (R / f).exists():
        bad = re.findall(r"^\s*(?:from|import)\s+webapp\b.*$", (R / f).read_text(encoding="utf-8", errors="ignore"), flags=re.M)
        if bad:
            ok = False; print(f"EDIT     {f}: still has  {bad[0].strip()}   <- change 'webapp' to 'api'")
        else: print("OK       " + f + " has no webapp import")

print("\n--- deploy_config.json keys")
cfgp = R / "models/deploy/deploy_config.json"
if cfgp.exists():
    cfg = json.load(open(cfgp))
    for k in ("classes", "img_size", "resize_policy", "cls_resize_policy", "seg_resize_policy", "cls_norm", "temperature", "bands", "seg_thr", "disclaimer"):
        good = k in cfg
        print(("OK       " if good else "MISSING  ") + k + ("" if good or not k.endswith("_policy") else '    <- add  "cls_resize_policy": "stretch",  "seg_resize_policy": "pad",  after "resize_policy"'))
        ok &= good

print("\n--- python packages")
for name, pip in (("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "opencv-python-headless"), ("PIL", "pillow"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                  ("tqdm", "tqdm"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("pytorch_grad_cam", "grad-cam")):
    try: importlib.import_module(name); print("OK       " + name)
    except Exception as e: print(f"MISSING  {name}    <- pip install {pip}   ({type(e).__name__})"); ok = False
try: importlib.import_module("python_multipart"); print("OK       python-multipart")
except Exception:
    try: importlib.import_module("multipart"); print("OK       python-multipart")
    except Exception: print("MISSING  python-multipart    <- pip install python-multipart"); ok = False

print("\n--- loading the models and the api app")
if ok:
    try:
        from local_app.pipeline import BrainTumorXAI
        eng = BrainTumorXAI(); print("OK       models loaded on", eng.device)
        from api.main import app
        print("OK       routes:", sorted({r.path for r in app.routes if getattr(r, "path", "").startswith("/api")}))
    except Exception:
        ok = False; print("FAILED   the full error is below - send it to me:\n"); traceback.print_exc()
else:
    print("skipped until the items above are fixed")

print("\nRESULT:", "READY. Run:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000   (Hugging Face: python app.py)" if ok else "NOT READY - fix the MISSING / EDIT / FAILED lines above, then run this again.")