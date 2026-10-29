#!/usr/bin/env python
"""Reproduction check (NOT a new evaluation): run the exported ONNX models on the cached BRISC test images and compare with the
predictions/Dice saved on Kaggle. Needs data/cache/*, outputs/tables/final_predictions_test.csv, models/deploy/*.

  python scripts/check_reproduction.py            # classification on all 1000 test images + segmentation on 200 tumor images
  python scripts/check_reproduction.py --seg-n 860
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score

ap = argparse.ArgumentParser(); ap.add_argument("--seg-n", type=int, default=200); ap.add_argument("--bs", type=int, default=32); a = ap.parse_args()
dep = ROOT / "models" / "deploy"; cfg = json.load(open(dep / "deploy_config.json")); classes = cfg["classes"]
ci = pd.read_csv(ROOT / "data/cache/cache_index_pad_256.csv")
X = np.load(ROOT / "data/cache/images_pad_256.npy", mmap_mode="r"); M = np.load(ROOT / "data/cache/masks_pad_256.npy", mmap_mode="r")
IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def batch(rows, mean, std):
    x = np.asarray(X[rows], np.float32)[:, None].repeat(3, 1) / 255.0
    return ((x - np.array(mean, np.float32).reshape(1, 3, 1, 1)) / np.array(std, np.float32).reshape(1, 3, 1, 1)).astype(np.float32)
cm = {"imagenet": IMN, "dataset": ((cfg["stats"][0],) * 3, (cfg["stats"][1],) * 3), "none": ((0, 0, 0), (1, 1, 1))}[cfg["cls_norm"]]

# ---- classification ----
cls = ort.InferenceSession(str(dep / "classifier.onnx"), providers=["CPUExecutionProvider"])
test = ci[ci.partition == "test"]; rows = test.index.values; logits = []
for i in range(0, len(rows), a.bs): logits.append(cls.run(["logits"], {"input": batch(rows[i:i + a.bs], *cm)})[0])
z = np.concatenate(logits) / cfg["temperature"]; pred = z.argmax(1)
true = test.tumor_label.map({c: i for i, c in enumerate(classes)}).values
saved = pd.read_csv(ROOT / "outputs/tables/final_predictions_test.csv").set_index("idx").loc[rows]
agree = (np.array([classes[k] for k in pred]) == saved["pred"].values).mean()
dd = test.test_dedup.values.astype(bool)
print(f"classification | agreement with saved Kaggle predictions: {agree * 100:.2f}%  (expect >= 99.5%)")
for name, m in (("official", np.ones(len(true), bool)), ("de-duplicated", dd)):
    print(f"   {name:14s} n={int(m.sum()):4d}  accuracy {accuracy_score(true[m], pred[m]):.4f}  macro-F1 {f1_score(true[m], pred[m], average='macro'):.4f}")
ok_cls = agree >= 0.995

# ---- segmentation ----
seg = ort.InferenceSession(str(dep / "segmenter.onnx"), providers=["CPUExecutionProvider"])
t = test[test.row_mask >= 0].sample(min(a.seg_n, int((test.row_mask >= 0).sum())), random_state=0); dice = []
for i in range(0, len(t), 16):
    r = t.iloc[i:i + 16]; p = 1 / (1 + np.exp(-seg.run(["mask_logits"], {"input": batch(r.index.values, *IMN)})[0][:, 0]))
    for pr, rm in zip(p, r.row_mask.values):
        pm, gt = pr > cfg["seg_thr"], np.asarray(M[int(rm)]) > 0; tot = pm.sum() + gt.sum()
        dice.append(2 * (pm & gt).sum() / tot if tot else 1.0)
sv = pd.read_csv(ROOT / "outputs/tables/final_seg_test_per_image.csv").set_index("index")
diff = np.abs(np.array(dice) - sv.loc[t.index.values, "dice"].values)
print(f"segmentation   | ONNX mean Dice {np.mean(dice):.4f} on {len(t)} images | mean |difference| to the saved Kaggle Dice: {diff.mean():.5f}  (expect < 0.005)")
ok_seg = diff.mean() < 0.005
print("\nREPRODUCTION CHECK:", "PASS" if ok_cls and ok_seg else "FAIL (see numbers above; the usual causes are a different cache, a different config, or a wrong ONNX export)")
sys.exit(0 if ok_cls and ok_seg else 1)