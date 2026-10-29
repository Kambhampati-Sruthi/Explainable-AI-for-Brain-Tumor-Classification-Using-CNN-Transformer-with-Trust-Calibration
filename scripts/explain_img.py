#!/usr/bin/env python
"""Run the WHOLE pipeline on an MRI slice (or a folder): calibrated classification -> segmentation -> explanations -> trust level.

  python scripts/explain_image.py samples/slice.png                              # fast: Grad-CAM (3 layers) + Guided Grad-CAM
  python scripts/explain_image.py samples/slice.png --methods all                # + Integrated Gradients + Occlusion (slow on CPU)
  python scripts/explain_image.py samples/slice.png --mask samples/slice_mask.png   # + localization scores against a ground-truth mask
  python scripts/explain_image.py samples/ --mask-dir samples_masks/ --out results/  # a folder; mask files share the image's file name stem

Outputs per image: results/<name>_explanation.png (figure) and results/<name>.json (numbers). Research prototype, not a medical device.
Explanation quality scores printed per method:
  deletion AUC (lower = more faithful)  insertion AUC (higher = more faithful)           [RISE-style, Petsiuk et al. 2018]
  with --mask: energy_in_mask, pointing_hit, iou_at_15pct (the Grad-CAM paper's 15%-of-max threshold)
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.deployment.inference import Engine
from src.deployment import layers as LY
from src.xai import methods as XM, evaluate as XE
from src.preprocessing.cache import process_mask


def explain_all(model, gmodel, x, t, base, which):
    """Heatmaps (256x256, non-negative, torch) for one image. Grad-CAM at three depths comes from ONE forward/backward pass."""
    maps, delta = {}, None
    mc = XM.MultiGradCAM(model, {"L8": model.features[8], "L5": model.features[5], "L3": model.features[3]})
    try: cams, _, _ = mc(x, target=t)
    finally: mc.remove()
    for k, g in (("L8", "8x8"), ("L5", "16x16"), ("L3", "32x32")):
        maps[f"Grad-CAM {k} ({g})"] = cams[k][0]
    if "guided" in which:                                                  # Grad-CAM x SiLU-adapted guided backprop
        maps["Guided Grad-CAM"] = (cams["L8"][:, None] * XM.guided_backprop(gmodel, x, t)).abs().sum(1)[0]
    if "ig" in which:                                                      # Captum Integrated Gradients, black baseline in normalised space
        attr, d = XM.ig_attr(model, x, t, base); delta = float(d.flatten()[0])
        maps["Integrated Gradients"] = attr.sum(1).clamp(min=0)[0].detach()
    if "occlusion" in which:                                               # Captum Occlusion, 32x32 window, stride 16
        maps["Occlusion"] = XM.occlusion_attr(model, x, t, base).mean(1).clamp(min=0)[0].detach()
    return maps, delta


def score_maps(model, x, t, base, maps, gt):
    rows = {}
    for name, h in maps.items():
        hn = h.detach().cpu().numpy(); d, i = XE.del_ins(model, x, h, t, base)
        r = {"deletion_auc": d, "insertion_auc": i}
        if gt is not None:
            r.update(energy_in_mask=XE.energy(hn, gt), pointing_hit=XE.pointing_hit(hn, gt), iou_at_15pct=XE.iou_at(hn, gt))
        rows[name] = {k: float(v) for k, v in r.items()}
    return rows


def draw(path, arr, p, s, level, reasons, maps, scores, gt, true_class, out_png, classes):
    names = list(maps); fig = plt.figure(figsize=(3.3 * max(len(names), 4), 8.2))
    gs = fig.add_gridspec(2, max(len(names), 4))
    a = fig.add_subplot(gs[0, 0]); a.imshow(arr, cmap="gray", vmin=0, vmax=255); a.axis("off")
    if s["mask"].any(): a.contour(s["mask"], levels=[0.5], colors="lime", linewidths=1.3)          # predicted tumor
    if gt is not None: a.contour(gt, levels=[0.5], colors="cyan", linewidths=1.3, linestyles="--")  # ground truth
    a.set_title("model view\ngreen = predicted tumor" + ("\ncyan dashed = ground truth" if gt is not None else ""), fontsize=9)
    b = fig.add_subplot(gs[0, 1]); order = np.argsort([p["probs"][c] for c in classes])
    b.barh([classes[i] for i in order], [p["probs"][classes[i]] for i in order]); b.set_xlim(0, 1); b.set_title("class probabilities (temperature-scaled)", fontsize=9)
    c = fig.add_subplot(gs[0, 2:]); c.axis("off")
    lines = [f"file: {Path(path).name}", f"PREDICTION: {p['cls']}  ({p['conf'] * 100:.1f}% confidence)",
             f"calibration band: {p['calibration']}", f"TRUST: {level}"] + ["  - " + r for r in reasons]
    lines += [f"segmentation: {s['px']} px ({s['pct']:.2f}% of slice)"]
    if true_class: lines.append(f"true class (given): {true_class}  ->  {'CORRECT' if true_class == p['cls'] else 'WRONG'}")
    lines.append("RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE")
    c.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="monospace")
    for j, n in enumerate(names):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(XM.overlay(arr, maps[n].cpu().numpy())); ax.axis("off")
        sc = scores[n]; t = f"{n}\ndel {sc['deletion_auc']:.2f} | ins {sc['insertion_auc']:.2f}"
        if gt is not None: t += f"\nenergy {sc['energy_in_mask']:.2f} | hit {int(sc['pointing_hit'])} | IoU {sc['iou_at_15pct']:.2f}"
        ax.set_title(t, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)


def run_one(path, engine, model, gmodel, dev, which, out_dir, mask_path=None, true_class=None):
    t0 = time.perf_counter(); cfg = engine.cfg
    im = Image.open(path); im.load()
    p, s = engine.predict(im), engine.segment(im)                         # ONNX: calibrated classification + segmentation
    arr, xc, _ = engine.prep(im)
    x, t = torch.from_numpy(xc).to(dev), p["k"]
    base = XM.black_baseline(cfg["cls_norm"], cfg["stats"], cfg["img_size"], dev)
    maps, delta = explain_all(model, gmodel, x, t, base, which)
    gt = None
    if mask_path:
        gt = process_mask(np.array(Image.open(mask_path).convert("L")), cfg["resize_policy"], cfg["img_size"]).astype(bool)
    scores = score_maps(model, x, t, base, maps, gt)
    tumor = p["cls"] != "no_tumor"
    consist = XE.energy(maps["Grad-CAM L8 (8x8)"].cpu().numpy(), s["mask"]) if tumor and s["px"] > 0 else None
    level, reasons, agree = LY.trust(p["conf"], tumor, s["px"] > 50, consist)
    stem = Path(path).stem; out_dir.mkdir(parents=True, exist_ok=True)
    draw(path, arr, p, s, level, reasons, maps, scores, gt, true_class, out_dir / f"{stem}_explanation.png", engine.classes)
    result = {"file": str(path), "prediction": {"class": p["cls"], "confidence": p["conf"], "probabilities": p["probs"], "calibration": p["calibration"]},
              "segmentation": {"area_px": s["px"], "area_percent": s["pct"]}, "trust": {"level": level, "reasons": reasons, "agreement": bool(agree), "consistency": consist},
              "explanations": scores, "ig_convergence_delta": delta, "seconds": round(time.perf_counter() - t0, 1), "disclaimer": cfg["disclaimer"]}
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
    print(f"\n{Path(path).name}: {p['cls']} ({p['conf'] * 100:.1f}%, {p['calibration']}) | trust {level} | tumor region {s['pct']:.2f}% | {result['seconds']} s")
    for n, sc in scores.items(): print(f"   {n:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in sc.items()))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image"); ap.add_argument("--deploy", default=str(ROOT / "models" / "deploy"))
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "deploy" / "classifier_best.pt"))
    ap.add_argument("--out", default="results"); ap.add_argument("--methods", default="guided", help="comma list of: guided, ig, occlusion, or 'all'")
    ap.add_argument("--mask"); ap.add_argument("--mask-dir"); ap.add_argument("--true-class", choices=["glioma", "meningioma", "no_tumor", "pituitary"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    for f in (Path(a.deploy) / "classifier.onnx", Path(a.deploy) / "segmenter.onnx", Path(a.deploy) / "deploy_config.json", Path(a.ckpt)):
        if not f.exists(): sys.exit(f"missing file: {f}\n(see the Git LFS / assemble steps; explanations need the PyTorch checkpoint)")
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    which = {"guided", "ig", "occlusion"} if a.methods == "all" else set(a.methods.split(","))
    engine = Engine(a.deploy, a.ckpt); model = LY._ensure_torch(engine).to(dev).eval()
    gmodel, n_swapped = XM.make_guided_model(model); print(f"device {dev} | SiLU layers adapted for guided backprop: {n_swapped}")
    src = Path(a.image); paths = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if src.is_dir() else [src]
    rows = []
    for p in paths:
        m = a.mask or (str(Path(a.mask_dir) / (p.stem + ".png")) if a.mask_dir and (Path(a.mask_dir) / (p.stem + ".png")).exists() else None)
        r = run_one(p, engine, model, gmodel, dev, which, Path(a.out), m, a.true_class); rows.append(r)
    if len(rows) > 1:
        import csv
        with open(Path(a.out) / "summary.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file", "class", "confidence", "calibration", "trust", "tumor_area_percent"])
            for r in rows: w.writerow([Path(r["file"]).name, r["prediction"]["class"], round(r["prediction"]["confidence"], 4), r["prediction"]["calibration"], r["trust"]["level"], round(r["segmentation"]["area_percent"], 3)])
        print("\nwrote", Path(a.out) / "summary.csv")


if __name__ == "__main__":
    main()