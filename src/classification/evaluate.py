"""Prediction and metrics.

Definitions (per class c, one-vs-rest):
  precision_c = TP / (TP + FP)    "of the images we called c, how many really were c"
  recall_c    = TP / (TP + FN)    "of the real c images, how many did we find"  (= sensitivity)
  F1_c        = 2 P R / (P + R)   harmonic mean of the two
  macro-F1    = unweighted mean of F1_c over classes (every class counts equally, regardless of its size)
  NLL         = -(1/N) sum_i log p_i[y_i]   (negative log-likelihood of the true class; punishes confident mistakes)
  AUC (OvR)   = area under the ROC curve of "class c vs the rest", averaged over classes
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)            # subtract the row max: avoids overflow in exp, result unchanged
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)         # each row now sums to 1


@torch.no_grad()                                    # no gradients needed: saves memory and time
def predict(model, loader, device):
    """Returns (logits (N,K) float32, labels (N,), cache_row_index (N,)) in loader order."""
    model.eval()                                    # BatchNorm uses running stats, Dropout off
    logits, labels, idx = [], [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(b["label"])
        idx.append(b["index"])
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(idx).numpy()


def metrics_from_logits(logits, labels, class_names):
    K = len(class_names)
    probs = softmax(logits.astype(np.float64))
    pred = probs.argmax(axis=1)
    p, r, f, sup = precision_recall_fscore_support(labels, pred, labels=list(range(K)), zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": sup}, index=class_names)
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:                              # a class is absent in this subset -> AUC undefined
        auc = float("nan")
    return {"accuracy": float((pred == labels).mean()), "macro_f1": float(f.mean()),
            "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
            "nll": nll, "auc_ovr": auc, "per_class": per_class,
            "confusion": confusion_matrix(labels, pred, labels=list(range(K))),
            "pred": pred, "probs": probs}


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (Brown, Cai & DasGupta 2001). Unlike the plain
    +-1.96*sqrt(p(1-p)/n) it behaves for small n and for p near 0 or 1, which is our situation."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half
