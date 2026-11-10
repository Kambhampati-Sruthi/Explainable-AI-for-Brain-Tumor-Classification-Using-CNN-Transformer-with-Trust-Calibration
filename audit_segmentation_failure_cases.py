"""Read-only deep-dive audit for selected BRISC2025 segmentation failures.

Run from the project root (the directory containing api/, src/, configs/, models/, dataset/):
    python audit_segmentation_failure_cases.py

This script does not train, tune thresholds, modify checkpoints, or change the app.
It uses the deployed ONNX segmenter and the repository's own preprocessing functions.
Outputs are written under report/segmentation_error_analysis/deep_dive/.
"""
from pathlib import Path
import csv
import json
import re
import sys
import traceback

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "dataset" / "brisc2025" / "segmentation_task" / "test" / "images"
MASK_DIR = ROOT / "dataset" / "brisc2025" / "segmentation_task" / "test" / "masks"
DEPLOY_DIR = ROOT / "models" / "deploy"
OUT_DIR = ROOT / "report" / "segmentation_error_analysis" / "deep_dive"
CASES = [
    "brisc2025_test_00001_gl_ax_t1",
    "brisc2025_test_00005_gl_ax_t1",
    "brisc2025_test_00045_gl_ax_t1",
    "brisc2025_test_00046_gl_ax_t1",
    "brisc2025_test_00910_pi_co_t1",
]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_file(folder, stem):
    for ext in EXTS:
        p = folder / (stem + ext)
        if p.is_file():
            return p
    return None


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def component_summary(mask):
    # 8-connected components; avoids adding a SciPy dependency.
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes = []
    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        count = 0
        while stack:
            cy, cx = stack.pop()
            count += 1
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(count)
    sizes.sort(reverse=True)
    return sizes[:10]


def main():
    required = [IMAGE_DIR, MASK_DIR, ROOT / "api" / "core.py",
                ROOT / "src" / "preprocessing" / "cache.py",
                DEPLOY_DIR / "deploy_config.json", DEPLOY_DIR / "segmenter.onnx"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required project paths:\n" + "\n".join(missing))

    sys.path.insert(0, str(ROOT))
    from api import core
    from src.preprocessing.cache import process_mask

    cfg = core.load_cfg(DEPLOY_DIR)
    size = int(cfg["img_size"])
    policy = cfg.get("seg_resize_policy", cfg["resize_policy"])
    threshold = float(cfg["seg_thr"])

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(DEPLOY_DIR / "segmenter.onnx"),
                                   sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("BRISC2025 SELECTED SEGMENTATION FAILURE AUDIT")
    print(f"Preprocessing: {policy}, {size}x{size}; deployed threshold: {threshold}")
    for stem in CASES:
        image_path = find_file(IMAGE_DIR, stem)
        mask_path = find_file(MASK_DIR, stem)
        if image_path is None or mask_path is None:
            rows.append({"case": stem, "error": f"missing image or mask: image={image_path}, mask={mask_path}"})
            print(f"SKIP {stem}: image or mask missing")
            continue

        try:
            with Image.open(image_path) as im:
                raw_img = np.asarray(im.convert("L"))
                prepared = core.prepare(im.convert("RGB"), cfg)
            with Image.open(mask_path) as mi:
                raw_mask = np.asarray(mi.convert("L"))

            x = np.asarray(prepared["xs"], dtype=np.float32)
            if x.ndim == 3:
                x = x[None, ...]
            logits = session.run(None, {input_name: x})[0]
            if logits.shape != (1, 1, size, size):
                raise ValueError(f"Unexpected ONNX output shape: {logits.shape}")
            # Numerically stable sigmoid.
            z = logits[0, 0].astype(np.float64)
            prob = np.empty_like(z)
            pos = z >= 0
            prob[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
            ez = np.exp(z[~pos])
            prob[~pos] = ez / (1.0 + ez)

            gt = process_mask(raw_mask, policy, size).astype(bool)
            pred = prob > threshold
            tp = int(np.logical_and(gt, pred).sum())
            fp = int(np.logical_and(~gt, pred).sum())
            fn = int(np.logical_and(gt, ~pred).sum())
            dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0

            # Image is displayed in the segmenter's padded frame.
            img_s = np.asarray(prepared["arr_s"], dtype=np.uint8)
            if img_s.shape != (size, size):
                raise ValueError(f"Unexpected preprocessed image shape: {img_s.shape}")

            raw_mask_values, raw_mask_counts = np.unique(raw_mask, return_counts=True)
            row = {
                "case": stem,
                "image_file": image_path.name,
                "mask_file": mask_path.name,
                "raw_image_width": int(raw_img.shape[1]),
                "raw_image_height": int(raw_img.shape[0]),
                "raw_mask_width": int(raw_mask.shape[1]),
                "raw_mask_height": int(raw_mask.shape[0]),
                "raw_dimensions_match": bool(raw_img.shape == raw_mask.shape),
                "raw_mask_unique_values": ";".join(map(str, raw_mask_values.tolist())),
                "raw_mask_foreground_pixels_gt127": int((raw_mask > 127).sum()),
                "processed_image_shape": str(img_s.shape),
                "processed_gt_pixels": int(gt.sum()),
                "processed_pred_pixels": int(pred.sum()),
                "tp": tp, "fp": fp, "fn": fn, "dice_recomputed": float(dice),
                "prob_min": float(prob.min()),
                "prob_mean": float(prob.mean()),
                "prob_median": float(np.median(prob)),
                "prob_p95": float(np.quantile(prob, .95)),
                "prob_p99": float(np.quantile(prob, .99)),
                "prob_max": float(prob.max()),
                "pred_threshold": threshold,
                "gt_bbox_xyxy": json.dumps(bbox(gt)),
                "pred_bbox_xyxy": json.dumps(bbox(pred)),
                "gt_largest_components": json.dumps(component_summary(gt)),
                "pred_largest_components": json.dumps(component_summary(pred)),
            }
            rows.append(row)

            # Overlay colours: GT-only=blue, prediction-only=red, overlap=green.
            overlay = np.repeat(img_s[..., None], 3, axis=2).astype(np.float32)
            overlay /= 255.0
            gt_only = gt & ~pred
            pred_only = pred & ~gt
            overlap = gt & pred
            overlay[gt_only] = [0.10, 0.35, 1.00]
            overlay[pred_only] = [1.00, 0.15, 0.15]
            overlay[overlap] = [0.10, 0.85, 0.25]

            fig, ax = plt.subplots(1, 5, figsize=(18, 4))
            ax[0].imshow(img_s, cmap="gray", vmin=0, vmax=255)
            ax[0].set_title(f"MRI padded {size}x{size}")
            ax[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            ax[1].set_title(f"Ground truth ({int(gt.sum())} px)")
            ax[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
            ax[2].set_title(f"Probability map\nmax={prob.max():.4f}, p99={np.quantile(prob,.99):.4f}")
            ax[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
            ax[3].set_title(f"Prediction > {threshold} ({int(pred.sum())} px)")
            ax[4].imshow(np.clip(overlay, 0, 1))
            ax[4].set_title(f"Overlay: Dice={dice:.4f}\nblue=FN red=FP green=TP")
            for a in ax:
                a.axis("off")
            fig.suptitle(f"{stem}\nraw MRI={raw_img.shape[1]}x{raw_img.shape[0]}, raw mask={raw_mask.shape[1]}x{raw_mask.shape[0]}")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{stem}_diagnostic.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"OK {stem}: raw dimensions match={row['raw_dimensions_match']}, "
                  f"GT={row['processed_gt_pixels']} px, pred={row['processed_pred_pixels']} px, "
                  f"max probability={row['prob_max']:.5f}, Dice={dice:.5f}")
        except Exception as exc:
            rows.append({"case": stem, "error": repr(exc)})
            print(f"ERROR {stem}: {exc}")

    csv_path = OUT_DIR / "selected_case_diagnostics.csv"
    if rows:
        columns = sorted({k for r in rows for k in r})
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    print("\nSaved diagnostic outputs under:")
    print(OUT_DIR)
    print(csv_path)
    print("\nInterpretation note: matching raw dimensions and filenames do not prove anatomical alignment.")
    print("Review the diagnostic overlays and probability maps before attributing failures to model behaviour.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
