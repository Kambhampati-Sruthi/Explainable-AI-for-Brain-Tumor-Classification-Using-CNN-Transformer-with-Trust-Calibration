"""Calibration tools. Notation: conf_i = max_k p_ik, correct_i = 1[argmax = y].
  ECE = sum_b (n_b/N) |acc_b - conf_b|   (15 equal-width bins; Guo et al. 2017)   MCE = max_b |acc_b - conf_b|
  Brier = (1/N) sum_i sum_k (p_ik - 1[y_i=k])^2
  Temperature scaling: p = softmax(z/T), one scalar T fitted on VALIDATION logits by minimising NLL.
  Risk-coverage: abstain on the least-confident images; risk(c) = error rate among the most-confident fraction c."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax as _sm, logsumexp

def softmax_T(z, T=1.0): return _sm(np.asarray(z, float) / T, axis=1)

def nll(z, y, T=1.0):
    z = np.asarray(z, float) / T
    return float(-(z[np.arange(len(y)), y] - logsumexp(z, axis=1)).mean())

def fit_temperature(z, y):
    r = minimize_scalar(lambda lt: nll(z, y, np.exp(lt)), bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(r.x))

def reliability_table(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)      # bin b = (edge_b, edge_{b+1}]
    rows = [(k, int((b == k).sum()), float(correct[b == k].mean()) if (b == k).any() else np.nan,
             float(conf[b == k].mean()) if (b == k).any() else np.nan) for k in range(n_bins)]
    return pd.DataFrame(rows, columns=["bin", "n", "acc", "conf"])

def ece_mce(conf, correct, n_bins=15):
    t = reliability_table(conf, correct, n_bins); t = t[t.n > 0]
    gap = (t.acc - t.conf).abs()
    return float((t.n / t.n.sum() * gap).sum()), float(gap.max())

def brier(p, y):
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return float(((p - oh) ** 2).sum(1).mean())

def risk_coverage(conf, correct):
    o = np.argsort(-conf, kind="stable"); err = 1 - correct[o]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return np.arange(1, len(err) + 1) / len(err), risk, float(risk.mean())      # last value = AURC

def build_bands(conf, correct, edges=(0.0, 0.5, 0.8, 0.95, 1.0001), min_n=20, tol=0.05):
    """Label each confidence band from VALIDATION data: Well-calibrated if |acc - mean conf| <= tol."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi); n = int(m.sum())
        acc, mc = (float(correct[m].mean()), float(conf[m].mean())) if n else (np.nan, np.nan)
        lab = "Insufficient data" if n < min_n else ("Well-calibrated" if abs(acc - mc) <= tol else
              ("Over-confident" if acc < mc else "Under-confident"))
        out.append(dict(lo=lo, hi=min(hi, 1.0), n=n, acc=acc, conf=mc, label=lab))
    return out

def band_label(conf, bands):
    for b in bands:
        if b["lo"] <= conf < b["hi"] or (conf >= 1.0 and b["hi"] >= 1.0):
            return b["label"]
    return "Insufficient data"
