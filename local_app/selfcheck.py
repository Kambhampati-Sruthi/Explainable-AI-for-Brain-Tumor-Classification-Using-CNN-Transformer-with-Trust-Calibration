"""One-minute check that your downloaded models work here:  python local_app/selfcheck.py"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
need = ["models/deploy/classifier_best.pt", "models/deploy/deploy_config.json", "configs/segmentation_primary.json"]
import json; prim = json.load(open(ROOT / "configs/segmentation_primary.json"))["primary"]; need.append(f"models/{prim}_best.pt")
miss = [n for n in need if not (ROOT / n).exists()]
if miss: sys.exit("MISSING files (copy them from your Kaggle download):\n  " + "\n  ".join(miss))
import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
try: import pytorch_grad_cam; print("grad-cam library OK", getattr(pytorch_grad_cam, "__version__", ""))
except ImportError: sys.exit("run: pip install grad-cam")
from local_app.pipeline import BrainTumorXAI, CAM_METHODS
eng = BrainTumorXAI(); print("models loaded on", eng.device)
rng = np.random.default_rng(0); img = (rng.random((512, 512)) * 40).astype(np.uint8); img[150:350, 150:350] += 80          # synthetic slice, only for shape/runtime checks
im = Image.fromarray(img)
for m in ("Grad-CAM", "Grad-CAM++"):
    t = time.perf_counter(); r = eng.analyze(im, m, 8, "lib"); assert r["cam"].shape == (256, 256) and 0 <= r["cam"].min() and r["cam"].max() <= 1.0001
    print(f"{m}: ok in {time.perf_counter() - t:.1f}s | pred {r['cls']} {r['conf']:.2f} | trust {r['trust']}" + (f" | lib-vs-own corr {r['lib_vs_own_corr']:.3f}" if r["lib_vs_own_corr"] is not None else ""))
print("SELF-CHECK PASSED (this used a synthetic image, so the prediction itself is meaningless)")