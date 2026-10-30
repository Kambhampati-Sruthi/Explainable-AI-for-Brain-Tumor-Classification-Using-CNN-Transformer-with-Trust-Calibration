#!/usr/bin/env python3
"""
evaluate_all.py  -  ONE file that recomputes every performance / validation metric of the brain-tumor project
                    from the result files your Kaggle notebook saved (outputs/tables/*).  No GPU, no dataset, no PyTorch.

    python evaluate_all.py                                   # auto-finds outputs/tables
    python evaluate_all.py --results path/to/outputs/tables --out report
    python evaluate_all.py --no-plots                        # numbers only

Needs:  pip install numpy scipy matplotlib        (matplotlib only for the figures)

What it produces (printed AND saved in --out):
  metrics_report.txt  metrics.json  *.csv  *.png
  1  Classifier (EfficientNet-B0): every seed x {validation, official test, de-duplicated test}: accuracy (+95% CI), macro-F1,
     macro precision/recall, AUC, NLL; mean +- std over seeds; 3-seed ensemble; per-class table and confusion matrix; wrong predictions
  2  Calibration, recomputed from the logits: temperature T per seed (fitted on validation), NLL / Brier / ECE raw vs scaled, risk-coverage
  3  Segmentation (U-Net++): Dice (+bootstrap CI), IoU, precision, recall, HD95, missed tumors; official vs de-duplicated; per tumor class
  4  Trust levels versus real errors
  5  Training curves summary (best epoch per run) and every other result table the notebook saved
  Figures: confusion matrices, ROC curves, reliability diagram, risk-coverage, training curves, Dice per class, trust levels
"""
import argparse, csv, json, math, os, re, sys, traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
SPLITS = [("validation", "Validation"), ("test_official", "Test (official)"), ("test_dedup", "Test (de-duplicated)")]
KEYS = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("macro_precision", "Macro prec."), ("macro_recall", "Macro recall"), ("auc", "AUC"), ("nll", "NLL")]


# =====================================================================================================  output helpers
class Out:
    def __init__(self): self.lines = []
    def __call__(self, s=""): print(s); self.lines.append(str(s))
    def h1(self, t): self(); self("=" * 108); self(t); self("=" * 108)
    def h2(self, t): self(); self(t); self("-" * len(t))


def f4(v): return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.4f}"
def pc(v, d=2): return "n/a" if v is None else f"{v * 100:.{d}f}%"
def pretty(s): return str(s).replace("_", " ")


def table(header, rows):
    rows = [[str(c) for c in r] for r in rows]; cols = list(zip(header, *rows)); w = [max(len(c) for c in col) for col in cols]
    line = lambda r: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


# =====================================================================================================  numeric helpers
def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion (stable near 0 and 1, unlike +-1.96*sqrt(p(1-p)/n))."""
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc_ovr(p, y):
    """Macro one-vs-rest ROC-AUC (Mann-Whitney rank formula). Classes absent from y are skipped."""
    a = []
    for c in range(p.shape[1]):
        pos = y == c; npos, nneg = int(pos.sum()), int((~pos).sum())
        if npos and nneg: a.append((rankdata(p[:, c])[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(a)) if a else None


def roc_curve(score, is_pos):
    o = np.argsort(-score, kind="mergesort"); s, b = score[o], is_pos[o].astype(float)
    tps, fps = np.cumsum(b), np.cumsum(1 - b); last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    return np.r_[0, fps[last] / max(fps[-1], 1)], np.r_[0, tps[last] / max(tps[-1], 1)]


def class_metrics(logits, y):
    """All scores of one model on one image subset. Confusion matrix rows = true class, columns = predicted class."""
    p = softmax(logits); pred = p.argmax(1); n, K = len(y), len(CLASSES)
    cm = np.zeros((K, K), int); np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float); sup, pn = cm.sum(1), cm.sum(0)
    pr, rc = tp / np.maximum(pn, 1), tp / np.maximum(sup, 1); f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return dict(n=n, accuracy=float(tp.sum() / n), accuracy_ci=wilson(int(tp.sum()), n), macro_f1=float(f1.mean()), macro_precision=float(pr.mean()),
                macro_recall=float(rc.mean()), auc=auc_ovr(p, y), nll=float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean()),
                errors=int(n - tp.sum()), precision=pr, recall=rc, f1=f1, support=sup, confusion=cm)


def fit_temperature(z, y):
    """One scalar T > 0 minimising validation NLL of softmax(z / T)  (Guo et al., 2017). T > 1 = the raw model was over-confident."""
    def nll(lt): p = softmax(z, np.exp(lt)); return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded").x))


def ece_mce(p, y, bins=15):
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float); edges = np.linspace(0, 1, bins + 1); idx = np.digitize(conf, edges[1:-1])
    gaps, rows = [], []
    for b in range(bins):
        m = idx == b
        if m.any(): gaps.append((m.mean(), abs(ok[m].mean() - conf[m].mean()))); rows.append((edges[b], edges[b + 1], int(m.sum()), float(ok[m].mean()), float(conf[m].mean())))
    return float(sum(w * g for w, g in gaps)), float(max((g for _, g in gaps), default=0)), rows


def brier(p, y): oh = np.eye(p.shape[1])[y]; return float(((p - oh) ** 2).sum(1).mean())


def risk_coverage(p, y):
    o = np.argsort(-p.max(1), kind="mergesort"); err = (p.argmax(1) != y)[o].astype(float); k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(err) / k


# =====================================================================================================  file readers
def read_csv(path):
    if not Path(path).exists(): return None
    with open(path, newline="") as f: return list(csv.DictReader(f))


def fnum(v):
    try: x = float(v); return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None


def flag(v): return str(v).strip().lower() in ("true", "1", "1.0")


def find_results(arg):
    for c in (arg, os.environ.get("BTX_RESULTS_DIR"), HERE / "outputs" / "tables", Path.cwd() / "outputs" / "tables", HERE, Path.cwd(), HERE.parent / "outputs" / "tables"):
        if c and any((Path(c) / n).exists() for n in ("final_s0_test_logits.npz", "final_predictions_test.csv", "final_seg_test_per_image.csv")): return Path(c)
    return None


def load_runs(d):
    preds = read_csv(d / "final_predictions_test.csv")
    dmap = {int(float(r["idx"])): flag(r["dedup"]) for r in preds} if preds else None          # which test images have NO twin in train
    runs = []
    for f in sorted(d.glob("final_s*_test_logits.npz")):
        m = re.fullmatch(r"final_s(\d+)_test_logits\.npz", f.name)
        if not m or not (d / f"final_s{m.group(1)}_val_logits.npz").exists(): continue
        v, t = np.load(d / f"final_s{m.group(1)}_val_logits.npz"), np.load(f); it = t["idx"].astype(int)
        runs.append(dict(seed=int(m.group(1)), lv=v["logits"].astype(np.float64), yv=v["labels"].astype(int), lt=t["logits"].astype(np.float64), yt=t["labels"].astype(int), it=it,
                         mask=np.array([dmap.get(int(i), True) for i in it]) if dmap else None))
    return runs, preds


def subset(r, split):
    if split == "validation": return r["lv"], r["yv"]
    if split == "test_official": return r["lt"], r["yt"]
    return (r["lt"][r["mask"]], r["yt"][r["mask"]]) if r["mask"] is not None else (None, None)


# =====================================================================================================  1. classification
def classification(d, out, res, outdir):
    runs, preds = load_runs(d)
    if not runs: out("SKIPPED: final_s*_val_logits.npz / final_s*_test_logits.npz not found."); return None
    M = {r["seed"]: {sp: class_metrics(*subset(r, sp)) for sp, _ in SPLITS if subset(r, sp)[0] is not None} for r in runs}
    primary = max(M, key=lambda s: M[s]["validation"]["macro_f1"])                       # chosen on VALIDATION only, as in the notebook
    splits = [sp for sp, _ in SPLITS if all(sp in M[s] for s in M)]; name = dict(SPLITS)
    out.h1("1. CLASSIFIER (EfficientNet-B0): accuracy and all scores")
    out(f"Seeds found: {sorted(M)} | primary seed (best validation macro-F1): {primary} | classes: {', '.join(CLASSES)}")
    out("Confusion matrices: rows = TRUE class, columns = PREDICTED class. 'De-duplicated' removes test images that also occur in the training set.")

    out.h2("1a. Every seed on every split")
    rows = []
    for s in sorted(M):
        for sp in splits:
            m = M[s][sp]; rows.append([f"seed {s}{' *' if s == primary else ''}", name[sp], m["n"], pc(m["accuracy"]), f"[{pc(m['accuracy_ci'][0], 1)}, {pc(m['accuracy_ci'][1], 1)}]"] + [f4(m[k]) for k, _ in KEYS[1:]] + [m["errors"]])
    out(table(["model", "split", "n", "accuracy", "95% CI", *[l for _, l in KEYS[1:]], "errors"], rows)); out("(* = primary seed)")

    out.h2("1b. Mean +- standard deviation over seeds, and the ensemble (average of the seeds' probabilities)")
    summ, rows = {}, []
    for sp in splits:
        summ[sp] = {k: (float(np.mean([M[s][sp][k] for s in M])), float(np.std([M[s][sp][k] for s in M], ddof=1)) if len(M) > 1 else None) for k, _ in KEYS}
        rows.append([f"mean of {len(M)} seeds", name[sp]] + [f4(summ[sp][k][0]) + (f" +- {summ[sp][k][1]:.4f}" if summ[sp][k][1] is not None else "") for k, _ in KEYS])
    ens = {}
    if len(M) > 1 and all(np.array_equal(runs[0]["it"], r["it"]) for r in runs):
        P = np.mean([softmax(r["lt"]) for r in runs], 0)
        for sp in ("test_official", "test_dedup"):
            if sp in splits:
                msk = np.ones(len(P), bool) if sp == "test_official" else runs[0]["mask"]; ens[sp] = class_metrics(np.log(np.clip(P[msk], 1e-12, None)), runs[0]["yt"][msk])
                rows.append(["ensemble", name[sp]] + [f4(ens[sp][k]) for k, _ in KEYS])
    out(table(["model", "split", *[l for _, l in KEYS]], rows))

    out.h2(f"1c. Primary seed {primary}: per-class scores and confusion matrix")
    for sp in splits:
        m = M[primary][sp]; out(f"\n{name[sp]}  (n={m['n']}, accuracy {pc(m['accuracy'])}, macro-F1 {f4(m['macro_f1'])}, errors {m['errors']})")
        out(table(["class", "precision", "recall", "F1", "support"], [[pretty(CLASSES[i]), f4(m["precision"][i]), f4(m["recall"][i]), f4(m["f1"][i]), int(m["support"][i])] for i in range(4)]))
        out(table(["true \\ predicted", *[pretty(c) for c in CLASSES]], [[pretty(CLASSES[i]), *map(int, m["confusion"][i])] for i in range(4)]))

    if preds:
        w = sorted((r for r in preds if not flag(r.get("correct"))), key=lambda r: -(fnum(r.get("conf")) or 0))
        out.h2(f"1d. Wrong predictions of the primary model on the official test set: {len(w)} (most confident mistakes first, up to 25)")
        out(table(["file", "true", "predicted", "confidence", "has twin in train?"], [[r.get("filename"), pretty(r.get("true")), pretty(r.get("pred")), f4(fnum(r.get("conf"))), "no" if flag(r.get("dedup")) else "YES"] for r in w[:25]]) if w else "none")

    # ---- files
    write_csv(outdir / "classification_all_runs.csv", ["seed", "split", "n", *[k for k, _ in KEYS], "errors"], [[s, sp, M[s][sp]["n"], *[M[s][sp][k] for k, _ in KEYS], M[s][sp]["errors"]] for s in sorted(M) for sp in splits])
    write_csv(outdir / "classification_summary.csv", ["split", *[f"{k}_{x}" for k, _ in KEYS for x in ("mean", "std")]], [[sp, *[v for k, _ in KEYS for v in summ[sp][k]]] for sp in splits])
    write_csv(outdir / "per_class_metrics.csv", ["seed", "split", "class", "precision", "recall", "f1", "support"], [[s, sp, CLASSES[i], M[s][sp]["precision"][i], M[s][sp]["recall"][i], M[s][sp]["f1"][i], int(M[s][sp]["support"][i])] for s in sorted(M) for sp in splits for i in range(4)])
    write_csv(outdir / "confusion_matrices.csv", ["seed", "split", "true", "predicted", "count"], [[s, sp, CLASSES[i], CLASSES[j], int(M[s][sp]["confusion"][i][j])] for s in sorted(M) for sp in splits for i in range(4) for j in range(4)])
    res["classification"] = dict(primary_seed=primary, seeds={s: {sp: M[s][sp] for sp in splits} for s in M}, summary=summ, ensemble=ens)
    return dict(runs=runs, M=M, primary=primary, splits=splits)


# =====================================================================================================  2. calibration
def calibration(d, out, res, outdir, C, seed=0):
    runs, primary, M = C["runs"], C["primary"], C["M"]; out.h1("2. CALIBRATION (recomputed from the logits)")
    out("Temperature T is fitted on VALIDATION logits only and then applied to the test logits. ECE uses 15 equal-width confidence bins.")
    rows, T, cal = [], {}, {}
    for r in runs:
        T[r["seed"]] = fit_temperature(r["lv"], r["yv"])
        for sp in ("test_official", "test_dedup"):
            z, y = subset(r, sp)
            if z is None: continue
            for tag, t in (("raw", 1.0), ("temperature-scaled", T[r["seed"]])):
                p = softmax(z, t); e, mce, _ = ece_mce(p, y); nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
                cal[(r["seed"], sp, tag)] = dict(nll=nll, brier=brier(p, y), ece=e, mce=mce)
                rows.append([f"seed {r['seed']}{' *' if r['seed'] == primary else ''}", dict(SPLITS)[sp], tag, f"{T[r['seed']]:.3f}" if tag != "raw" else "1", f4(nll), f4(brier(p, y)), f4(e), f4(mce)])
    out(table(["model", "split", "confidence", "T", "NLL", "Brier", "ECE", "MCE"], rows))
    out("\nT above 1 means the raw network was over-confident. Scaling never changes the predicted class, only the confidence.")
    prim = next(r for r in runs if r["seed"] == primary); z, y = subset(prim, "test_official"); rng = np.random.default_rng(seed)
    for tag, t in (("raw", 1.0), ("temperature-scaled", T[primary])):
        p = softmax(z, t); b = [ece_mce(softmax(z[i], t), y[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
        out(f"Primary seed {primary}, official test, {tag}: ECE {ece_mce(p, y)[0]:.4f}  bootstrap 95% [{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]")
    if (np.argmax(softmax(z), 1) == y).mean() > 0.98: out("Note: at this accuracy almost every image falls in the top confidence bin, so ECE rests on a few images; read NLL and the bootstrap interval, not ECE alone.")
    pt = softmax(z, T[primary]); cov, risk = risk_coverage(pt, y); out(f"\nRisk-coverage (primary, temperature-scaled): " + " | ".join(f"keep {c:.0%}: error {risk[int(round(c * len(cov))) - 1]:.4f}" for c in (0.90, 0.95, 0.99, 1.0)))
    write_csv(outdir / "calibration_recomputed.csv", ["seed", "split", "confidence", "T", "nll", "brier", "ece", "mce"], [[s, sp, tg, (T[s] if tg != "raw" else 1), v["nll"], v["brier"], v["ece"], v["mce"]] for (s, sp, tg), v in cal.items()])
    res["calibration"] = dict(T=T, rows={f"{s}|{sp}|{tg}": v for (s, sp, tg), v in cal.items()})
    saved = None
    try: saved = json.load(open(d / "calibration.json"))
    except Exception: pass
    if saved and "T" in saved: out(f"Temperature stored by the notebook (configs/calibration.json): T = {saved['T']:.3f} for seed {saved.get('primary_seed')}; recomputed here: {T.get(saved.get('primary_seed'), float('nan')):.3f}")
    C["T"] = T


# =====================================================================================================  3. segmentation
def segmentation(d, out, res, outdir):
    rows = read_csv(d / "final_seg_test_per_image.csv")
    if not rows: out("\nSKIPPED segmentation: final_seg_test_per_image.csv not found."); return None
    g = lambda k: np.array([(fnum(r.get(k)) if fnum(r.get(k)) is not None else np.nan) for r in rows])
    gt, pp, dice, iou, prec, rec, hd = g("gt_px"), g("pred_px"), g("dice"), g("iou"), g("precision"), g("recall"), g("hd95")
    ded, lab = np.array([flag(r.get("test_dedup")) for r in rows]), np.array([r.get("tumor_label", "") for r in rows]); rng = np.random.default_rng(0)
    def summ(m):
        m = m & (gt > 0); x = dice[m]
        if not len(x): return None
        bs = rng.choice(x, size=(2000, len(x)), replace=True).mean(1)
        return dict(n=int(m.sum()), dice=float(x.mean()), ci=tuple(np.percentile(bs, [2.5, 97.5])), iou=float(np.nanmean(iou[m])), precision=float(np.nanmean(prec[m])), recall=float(np.nanmean(rec[m])),
                    hd95_median=float(np.nanmedian(hd[m])) if np.isfinite(hd[m]).any() else None, missed=int((pp[m] == 0).sum()))
    out.h1("3. SEGMENTATION (U-Net++): tumor images only, primary model, test set")
    out("Dice = 2|A and B| / (|A|+|B|); IoU = |A and B| / |A or B|; HD95 = 95th-percentile boundary distance in pixels at 256 px; 'missed' = the model predicted no tumor pixel.")
    allm, S, lines, csvrows = np.ones(len(rows), bool), {}, [], []
    for view, vm in (("official test", allm), ("de-duplicated test", ded)):
        s = summ(vm); S[view] = s
        if s:
            lines.append([view, s["n"], f4(s["dice"]), f"[{f4(s['ci'][0])}, {f4(s['ci'][1])}]", f4(s["iou"]), f4(s["precision"]), f4(s["recall"]), f4(s["hd95_median"]), s["missed"]])
            csvrows.append([view, "all", s["n"], s["dice"], s["ci"][0], s["ci"][1], s["iou"], s["precision"], s["recall"], s["hd95_median"], s["missed"]])
    out(table(["view", "n", "Dice", "95% CI (bootstrap)", "IoU", "precision", "recall", "HD95 median", "missed"], lines))
    out.h2("Per tumor class"); pcl = []
    for view, vm in (("official test", allm), ("de-duplicated test", ded)):
        for c in sorted(set(lab[(gt > 0) & vm].tolist())):
            s = summ(vm & (lab == c))
            if s:
                pcl.append([view, pretty(c), s["n"], f4(s["dice"]), f"[{f4(s['ci'][0])}, {f4(s['ci'][1])}]", f4(s["iou"]), f4(s["hd95_median"]), s["missed"]])
                csvrows.append([view, c, s["n"], s["dice"], s["ci"][0], s["ci"][1], s["iou"], s["precision"], s["recall"], s["hd95_median"], s["missed"]])
    out(table(["view", "class", "n", "Dice", "95% CI", "IoU", "HD95 median", "missed"], pcl))
    write_csv(outdir / "segmentation_summary.csv", ["view", "class", "n", "dice", "dice_ci_low", "dice_ci_high", "iou", "precision", "recall", "hd95_median", "missed"], csvrows)
    res["segmentation"] = {k: v for k, v in S.items()}
    return dict(lab=lab, dice=dice, gt=gt)


# =====================================================================================================  4. trust
def trust(d, out, res, outdir):
    rows = read_csv(d / "trust_signals_test.csv")
    if not rows: out("\nSKIPPED trust levels: trust_signals_test.csv not found."); return None
    def level(r):                                                                                       # the notebook's Stage 6 rule, thresholds fixed in advance
        if "trust" in r and r["trust"] in ("HIGH", "MEDIUM", "LOW"): return r["trust"]
        conf, agree, tum, e = fnum(r["conf"]), flag(r["agree"]), flag(r["tum_pred"]), fnum(r.get("e_pred_L8"))
        return "LOW" if (conf < 0.80 or not agree) else "MEDIUM" if (conf < 0.95 or (tum and e is not None and e < 0.30)) else "HIGH"
    out.h1("4. TRUST LEVELS versus real classification errors (test set)")
    out("Rule: LOW if calibrated confidence < 0.80 or classifier and segmenter disagree; MEDIUM if confidence < 0.95 or the heat-map lies mostly outside the region; else HIGH.")
    res_rows, table_rows = {}, []
    for view in ("official", "dedup"):
        sub = [r for r in rows if view == "official" or flag(r.get("dedup"))]
        for L in ("HIGH", "MEDIUM", "LOW"):
            gset = [r for r in sub if level(r) == L]; k = sum(1 for r in gset if not flag(r["correct"])); lo, hi = wilson(k, len(gset))
            res_rows[f"{view}|{L}"] = dict(n=len(gset), errors=k, error_rate=k / len(gset) if gset else None)
            table_rows.append([view, L, len(gset), k, pc(k / len(gset), 2) if gset else "n/a", f"[{pc(lo, 1)}, {pc(hi, 1)}]" if lo is not None else "n/a"])
    out(table(["test view", "trust level", "images", "errors", "error rate", "Wilson 95% interval"], table_rows))
    out(f"Classifier and segmenter agree on {pc(np.mean([flag(r['agree']) for r in rows]), 1)} of test images. The total number of errors is small, so these intervals are wide.")
    write_csv(outdir / "trust_levels.csv", ["view", "level", "n", "errors", "error_rate", "wilson_95"], table_rows); res["trust"] = res_rows
    return res_rows


# =====================================================================================================  5. history + other tables
def history_and_tables(d, out, res, outdir):
    out.h1("5. TRAINING HISTORY and the other result tables saved by the notebook")
    hist = {}
    for f in sorted(d.glob("*_history.csv")):
        rows = read_csv(f)
        if rows: hist[f.name.replace("_history.csv", "")] = rows
    lines = []
    for n, rows in hist.items():
        col = "val_macro_f1" if "val_macro_f1" in rows[0] else "val_dice" if "val_dice" in rows[0] else None
        if not col: continue
        best = max(rows, key=lambda r: fnum(r[col]) or -1)
        lines.append([n, len(rows), "classifier" if col == "val_macro_f1" else "segmenter", col, f4(fnum(best[col])), best.get("epoch"), best.get("phase"), f4(fnum(rows[-1].get("train_loss")))])
    if lines: out(table(["run", "epochs run", "task", "selection metric", "best value", "best epoch", "phase", "last train loss"], lines))
    for fn, title in (("table_3_test_results.csv", "Notebook table 3: classifier test results"), ("table_4_test_results.csv", "Notebook table 4: segmentation test results"), ("table_6_calibration.csv", "Notebook table 6: calibration"),
                      ("final_format_shift_test.csv", "Format-shift test (does the model rely on file format?)"), ("table_3_efficiency.csv", "Classifier cost"), ("table_4_efficiency.csv", "Segmenter cost"),
                      ("seg_ablation_results.csv", "Segmentation ablation runs"), ("ablation_results.csv", "Classification ablation runs")):
        rows = read_csv(d / fn)
        if not rows: continue
        keys = list(rows[0]); out.h2(f"{title}   [{fn}]")
        out(table([pretty(k) for k in keys], [[(f"{fnum(r[k]):.4f}" if fnum(r[k]) is not None and not re.fullmatch(r"-?\d+", r[k].strip()) else r[k]) for k in keys] for r in rows[:40]]))
        if len(rows) > 40: out(f"... {len(rows) - 40} more rows in the file")
    return hist


# =====================================================================================================  figures
def figures(outdir, dpi, C, hist, seg, tr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def done(n): print(f"  saved {n}")
    if C:
        runs, M, primary, splits = C["runs"], C["M"], C["primary"], C["splits"]; name = dict(SPLITS); seeds = sorted(M)
        fig, ax = plt.subplots(len(seeds), len(splits), figsize=(4.6 * len(splits), 4.2 * len(seeds)), squeeze=False)
        for i, s in enumerate(seeds):
            for j, sp in enumerate(splits):
                cm = M[s][sp]["confusion"]; a = ax[i][j]; a.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
                a.set_title(f"seed {s}{' (primary)' if s == primary else ''} | {name[sp]}\nacc {M[s][sp]['accuracy']:.4f}  macro-F1 {M[s][sp]['macro_f1']:.4f}", fontsize=9)
                a.set_xticks(range(4)); a.set_xticklabels([pretty(c) for c in CLASSES], rotation=40, ha="right", fontsize=8); a.set_yticks(range(4)); a.set_yticklabels([pretty(c) for c in CLASSES], fontsize=8)
                if i == len(seeds) - 1: a.set_xlabel("predicted")
                if j == 0: a.set_ylabel("true")
                for p in range(4):
                    for q in range(4): a.text(q, p, int(cm[p, q]), ha="center", va="center", fontsize=9, color="white" if cm[p, q] / max(cm[p].sum(), 1) > 0.5 else "black")
        fig.tight_layout(); fig.savefig(outdir / "confusion_matrices.png", dpi=dpi); plt.close(fig); done("confusion_matrices.png")
        prim = next(r for r in runs if r["seed"] == primary)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        for a, sp in zip(ax, ("test_official", "test_dedup")):
            z, y = subset(prim, sp)
            if z is None: a.axis("off"); continue
            p = softmax(z)
            for c in range(4):
                if (y == c).any(): fpr, tpr = roc_curve(p[:, c], y == c); a.plot(fpr, tpr, label=f"{pretty(CLASSES[c])} (AUC {np.trapezoid(tpr, fpr):.4f})")
            a.plot([0, 1], [0, 1], "k:", lw=1); a.set_xlabel("false-positive rate"); a.set_ylabel("true-positive rate"); a.set_title(f"ROC one-vs-rest | seed {primary} | {name[sp]}", fontsize=10); a.legend(loc="lower right", fontsize=8); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(outdir / "roc_curves.png", dpi=dpi); plt.close(fig); done("roc_curves.png")
        if "T" in C:
            z, y = subset(prim, "test_official"); fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
            for tag, t, col in (("raw", 1.0, "tab:red"), ("temperature-scaled", C["T"][primary], "tab:blue")):
                p = softmax(z, t); e, _, rows_ = ece_mce(p, y); ax[0].plot([(r[4]) for r in rows_], [r[3] for r in rows_], "o-", color=col, label=f"{tag} (ECE {e:.4f})")
                cov, risk = risk_coverage(p, y); ax[1].plot(cov, risk, color=col, label=tag)
            ax[0].plot([0, 1], [0, 1], "k:", label="perfect"); ax[0].set_xlabel("mean confidence in bin"); ax[0].set_ylabel("accuracy in bin"); ax[0].set_title(f"Reliability diagram | seed {primary} | official test", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
            ax[1].set_xlabel("coverage (share of images kept, most confident first)"); ax[1].set_ylabel("error rate among kept images"); ax[1].set_title("Risk-coverage", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
            fig.tight_layout(); fig.savefig(outdir / "calibration.png", dpi=dpi); plt.close(fig); done("calibration.png")
    cls = {n: r for n, r in (hist or {}).items() if r and "val_macro_f1" in r[0]}; sg = {n: r for n, r in (hist or {}).items() if r and "val_dice" in r[0]}
    for group, col, loss, fn, title in ((cls, "val_macro_f1", "val_nll", "training_curves_classifier.png", "Classifier"), (sg, "val_dice", "val_loss", "training_curves_segmenter.png", "Segmenter")):
        if not group: continue
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for n, rows in group.items():
            ep = [fnum(r["epoch"]) for r in rows]; ax[0].plot(ep, [fnum(r[col]) for r in rows], label=n); ax[1].plot(ep, [fnum(r.get(loss)) for r in rows], label=n)
        ax[0].set_title(f"{title}: validation {col.replace('val_', '')}"); ax[1].set_title(f"{title}: validation {loss.replace('val_', '')}"); [(a.set_xlabel("epoch"), a.grid(alpha=.3), a.legend(fontsize=8)) for a in ax]
        fig.tight_layout(); fig.savefig(outdir / fn, dpi=dpi); plt.close(fig); done(fn)
    if seg is not None:
        fig, a = plt.subplots(figsize=(6.5, 4.4)); m = seg["gt"] > 0; cs = sorted(set(seg["lab"][m].tolist()) - {""})
        a.boxplot([seg["dice"][m & (seg["lab"] == c)] for c in cs], labels=[pretty(c) for c in cs], showmeans=True); a.set_ylabel("Dice per image"); a.set_title("Segmentation Dice by tumor class (official test)", fontsize=10); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(outdir / "segmentation_dice_by_class.png", dpi=dpi); plt.close(fig); done("segmentation_dice_by_class.png")
    if tr:
        fig, a = plt.subplots(figsize=(6, 4)); ls = ("HIGH", "MEDIUM", "LOW"); a.bar(ls, [100 * (tr[f"official|{L}"]["error_rate"] or 0) for L in ls], color=["tab:green", "tab:orange", "tab:red"])
        for i, L in enumerate(ls): a.text(i, 100 * (tr[f"official|{L}"]["error_rate"] or 0), f"{tr[f'official|{L}']['errors']}/{tr[f'official|{L}']['n']}", ha="center", va="bottom")
        a.set_ylabel("classification error rate (%)"); a.set_title("Error rate by trust level (official test)", fontsize=10); fig.tight_layout(); fig.savefig(outdir / "trust_levels.png", dpi=dpi); plt.close(fig); done("trust_levels.png")


# =====================================================================================================  main
def main():
    ap = argparse.ArgumentParser(description="Recompute all performance and validation metrics from the saved notebook results.")
    ap.add_argument("--results", help="folder with final_s*_*_logits.npz and the CSV tables (default: auto-detect outputs/tables)")
    ap.add_argument("--out", default="metrics_report", help="folder for the report, CSVs and figures")
    ap.add_argument("--no-plots", action="store_true"); ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(); d = find_results(a.results)
    if d is None: sys.exit("Result files not found. Copy the notebook's outputs/tables folder next to this script (as outputs/tables) or pass --results PATH.")
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True); out, res = Out(), {}
    out("Brain tumor XAI project: performance and validation metrics"); out(f"results folder: {d}"); out(f"output folder : {outdir.resolve()}")
    out("Research prototype. Not a medical device. The test set was evaluated once in the notebook; nothing here was tuned on it.")
    C = seg = tr = hist = None
    def step(fn, *args):
        try: return fn(*args)
        except Exception as e: out(f"\n!! {fn.__name__} failed: {type(e).__name__}: {e}"); traceback.print_exc(); return None
    C = step(classification, d, out, res, outdir)
    if C: step(calibration, d, out, res, outdir, C)
    seg = step(segmentation, d, out, res, outdir); tr = step(trust, d, out, res, outdir); hist = step(history_and_tables, d, out, res, outdir)
    (outdir / "metrics_report.txt").write_text("\n".join(out.lines), encoding="utf-8")
    json.dump(clean(res), open(outdir / "metrics.json", "w"), indent=2, allow_nan=False)
    if not a.no_plots:
        try: print("\nFigures:"); figures(outdir, a.dpi, C, hist, seg, tr)
        except ImportError: print("matplotlib is not installed: skipped the figures (pip install matplotlib).")
        except Exception as e: print(f"figures failed: {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"\nDone. Everything is in: {outdir.resolve()}")


if __name__ == "__main__":
    main()