"""Re-evaluate the trained classifier from the ORIGINAL dataset images (not from cached arrays or saved logits).

Run from the project root:
    python evaluate_test.py --data "C:\\path\\to\\brisc2025"            # test set only (about 1-3 minutes on a CPU)
    python evaluate_test.py --data "C:\\path\\to\\brisc2025" --extra --leakage

--data     folder that contains manifest.csv (or set BTX_DATA_ROOT)
--extra    also evaluate the validation set and a fixed 1,000-image sample of the training set (overfitting check)
--leakage  count test images that are byte-identical to a training or validation image
--backend  torch (default, uses models/deploy/classifier_best.pt) or onnx (uses models/deploy/classifier.onnx, no PyTorch needed)
--limit N  evaluate only N images per partition (quick test)
Results are written to outputs/reeval/.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from local_app import core

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("BTX_DATA_ROOT"))
ap.add_argument("--backend", choices=("torch", "onnx"), default="torch")
ap.add_argument("--ckpt", default=str(ROOT / "models/deploy/classifier_best.pt"))
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--extra", action="store_true")
ap.add_argument("--leakage", action="store_true")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
if not a.data: sys.exit("give the dataset folder:  --data \"D:\\path\\to\\brisc2025\"   (the folder that contains manifest.csv)")
DATA = Path(a.data)
if not (DATA / "manifest.csv").exists(): sys.exit(f"manifest.csv not found in {DATA}")

cfg = core.load_cfg(ROOT / "models/deploy"); CLASSES = cfg["classes"]; C2I = {c: i for i, c in enumerate(CLASSES)}
OUT = ROOT / "outputs" / "reeval"; OUT.mkdir(parents=True, exist_ok=True)

# ---------- files: image path of every classification image ----------
man = pd.read_csv(DATA / "manifest.csv")
man["abs"] = str(DATA) + "/" + man["relative_path"].str.replace("\\", "/", regex=False)
paths = man[(man["task"] == "classification") & (~man["is_mask"])].set_index("filename")["abs"]
sp = pd.read_csv(ROOT / "data/splits.csv")
missing = sorted(set(sp["filename"]) - set(paths.index))
if missing: sys.exit(f"{len(missing)} files from splits.csv are not in the manifest, e.g. {missing[:3]}")
sp["path"] = sp["filename"].map(paths)

# ---------- model ----------
if a.backend == "torch":
    import torch
    from src.classification.model import build_efficientnet_b0
    model = build_efficientnet_b0(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]); model.eval()
    def logits_of(x):
        with torch.no_grad(): return model(torch.from_numpy(x)).float().numpy()
else:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ROOT / "models/deploy/classifier.onnx"), providers=["CPUExecutionProvider"]); iname = sess.get_inputs()[0].name
    def logits_of(x): return sess.run(None, {iname: x})[0]


def predict(rows, label):
    """Original image files -> exactly the preprocessing the app uses -> logits."""
    out, t0 = [], time.perf_counter()
    for i in range(0, len(rows), a.batch):
        xs = []
        for p in rows["path"].values[i:i + a.batch]:
            with Image.open(p) as im: xs.append(core.prepare(im, cfg)["xc"])
        out.append(logits_of(np.concatenate(xs, 0)))
        print(f"\r  {label}: {min(i + a.batch, len(rows))}/{len(rows)}", end="", flush=True)
    print(f"  ({time.perf_counter() - t0:.0f} s)")
    return np.concatenate(out, 0)


# ---------- metrics (numpy only) ----------
def confusion(y, p, k=4):
    cm = np.zeros((k, k), int)
    for t, q in zip(y, p): cm[t, q] += 1
    return cm


def report(y, p):
    cm = confusion(y, p); tp = np.diag(cm)
    prec, rec = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return cm, prec, rec, f1, float((y == p).mean())


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(correct, n=2000, seed=0):
    r = np.random.default_rng(seed); b = r.choice(correct.astype(float), (n, len(correct))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def show(name, df, logits):
    y = df["tumor_label"].map(C2I).values; p = logits.argmax(1); cm, prec, rec, f1, acc = report(y, p)
    lo, hi = wilson(int((y == p).sum()), len(y)); blo, bhi = boot_ci(y == p)
    print(f"\n=== {name}: n={len(y)} ===")
    print(f"accuracy {acc * 100:.2f}%  (Wilson 95% CI {lo * 100:.1f}-{hi * 100:.1f}%, bootstrap {blo * 100:.1f}-{bhi * 100:.1f}%) | errors {int((y != p).sum())} | macro-F1 {f1.mean():.4f}")
    print(pd.DataFrame({"class": CLASSES, "support": cm.sum(1), "precision": prec.round(4), "recall": rec.round(4), "F1": f1.round(4)}).to_string(index=False))
    print("confusion matrix (rows = true, columns = predicted):"); print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    return dict(n=len(y), accuracy=acc, errors=int((y != p).sum()), macro_f1=float(f1.mean()), ci_wilson=[lo, hi], ci_bootstrap=[float(blo), float(bhi)])


summary = {}
# ---------- 1. test set ----------
test = sp[sp["partition"] == "test"].sort_values("filename").reset_index(drop=True)
if a.limit: test = test.groupby("tumor_label", group_keys=False).head(max(1, a.limit // 4)).reset_index(drop=True)
print(f"Evaluating {len(test)} test images with backend '{a.backend}' ...")
lt = predict(test, "test")
summary["test_official"] = show("TEST, official", test, lt)
dd = test["test_dedup"].astype(bool).values
if dd.any() and not dd.all(): summary["test_dedup"] = show("TEST, de-duplicated (images with a near-duplicate twin in training removed)", test[dd].reset_index(drop=True), lt[dd])
pr = lt.argmax(1); y = test["tumor_label"].map(C2I).values
pd.DataFrame({"filename": test["filename"], "true": test["tumor_label"], "pred": [CLASSES[i] for i in pr], "correct": pr == y, "test_dedup": dd}).to_csv(OUT / "test_predictions_reeval.csv", index=False)

# ---------- 2. does the local pipeline reproduce the notebook? ----------
npz, pcsv = ROOT / "outputs/tables/final_s0_test_logits.npz", ROOT / "outputs/tables/final_predictions_test.csv"
if npz.exists() and pcsv.exists():
    z = np.load(npz); fn = pd.read_csv(pcsv).set_index("idx")["filename"]
    saved = dict(zip([fn[i] for i in z["idx"]], z["logits"].argmax(1)))
    mine = dict(zip(test["filename"], pr)); common = [f for f in mine if f in saved]
    if common:
        agree = float(np.mean([saved[f] == mine[f] for f in common]))
        print(f"\nAgreement with the notebook's saved predictions (seed 0): {agree * 100:.2f}% on {len(common)} images")
        summary["agreement_with_notebook_seed0"] = agree

# ---------- 3. overfitting check ----------
if a.extra:
    val = sp[sp["partition"] == "val"].sort_values("filename").reset_index(drop=True)
    trn = sp[sp["partition"] == "train"].sample(n=min(1000, (sp["partition"] == "train").sum()), random_state=0).reset_index(drop=True)
    if a.limit: val, trn = val.head(a.limit), trn.head(a.limit)
    summary["validation"] = show("VALIDATION", val, predict(val, "validation"))
    summary["train_sample"] = show("TRAIN (random sample of 1000, no augmentation)", trn, predict(trn, "train sample"))
    print(f"\ntrain {summary['train_sample']['accuracy'] * 100:.2f}% | validation {summary['validation']['accuracy'] * 100:.2f}% | test {summary['test_official']['accuracy'] * 100:.2f}%")

# ---------- 4. exact-duplicate leakage check ----------
if a.leakage:
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
        return h.hexdigest()
    print("\nHashing files for the exact-duplicate check ...")
    sp["sha"] = [sha(p) for p in sp["path"]]
    seen = set(sp.loc[sp["partition"].isin(["train", "val"]), "sha"]); tst = sp[sp["partition"] == "test"]
    dup = int(tst["sha"].isin(seen).sum())
    print(f"test images that are byte-identical to a training or validation image: {dup} of {len(tst)}")
    print(f"duplicate files inside the test set itself: {int(tst['sha'].duplicated().sum())}")
    summary["exact_duplicates_test_in_train_or_val"] = dup

json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
print(f"\nSaved: {OUT / 'summary.json'} and {OUT / 'test_predictions_reeval.csv'}")