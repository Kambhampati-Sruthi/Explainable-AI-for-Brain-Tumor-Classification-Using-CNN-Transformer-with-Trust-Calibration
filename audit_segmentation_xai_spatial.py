
"""
Quantitative spatial audit of segmentation explanations.

Run from repository root:
    python audit_segmentation_xai_spatial.py

Run one case:
    python audit_segmentation_xai_spatial.py --case brisc2025_test_00045_gl_ax_t1

Run all selected cases:
    python audit_segmentation_xai_spatial.py

Outputs:
    report/segmentation_xai_spatial/
        raw_maps/
        segmentation_xai_spatial_metrics.csv

IMPORTANT:
Ground-truth-targeted explanations are offline research analyses only.
These metrics measure spatial correspondence, not clinical validity.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.segmentation.model import UNetPlusPlus, EffB1Encoder
from local_app import core
from segmentation_xai import SegmentationXAI


CASES = [
    "brisc2025_test_00001_gl_ax_t1",
    "brisc2025_test_00005_gl_ax_t1",
    "brisc2025_test_00045_gl_ax_t1",
    "brisc2025_test_00046_gl_ax_t1",
    "brisc2025_test_00910_pi_co_t1",
]


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_model(device):

    deploy_dir = ROOT / "models" / "deploy"
    cfg = core.load_cfg(deploy_dir)

    primary_config = json.loads(
        (ROOT / "configs" / "segmentation_primary.json").read_text(
            encoding="utf-8"
        )
    )

    primary = primary_config["primary"]

    checkpoint_path = (
        ROOT / "models" / f"{primary}_best.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_cfg = checkpoint.get("cfg", {})

    model = UNetPlusPlus(
        pretrained=False,
        nested=model_cfg.get("nested", True)
    )

    if model_cfg.get("variant", "b0") == "b1":
        model.encoder = EffB1Encoder(pretrained=False)

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, cfg, checkpoint_path


# ---------------------------------------------------------
# ARRAY UTILITIES
# ---------------------------------------------------------

def as_2d(array, name):

    array = np.asarray(array, dtype=np.float64)

    # Remove singleton dimensions only.
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D map; received shape {array.shape}"
        )

    return np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


def normalize01(array):

    array = as_2d(array, "map")

    lo = float(array.min())
    hi = float(array.max())

    if hi - lo < 1e-12:
        return np.zeros_like(array, dtype=np.float32)

    return ((array - lo) / (hi - lo)).astype(np.float32)


def rank_auc(scores, positive_mask):
    """
    Mann-Whitney rank AUC.

    Measures whether scores tend to be higher inside the
    ground-truth tumour than outside it.

    AUC = 0.5 means no rank separation.
    AUC > 0.5 means higher scores tend to occur inside GT.
    AUC < 0.5 means higher scores tend to occur outside GT.

    This is a spatial separation statistic, not an explanation
    faithfulness metric.
    """

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(positive_mask, dtype=bool).ravel()

    valid = np.isfinite(scores)

    scores = scores[valid]
    labels = labels[valid]

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks = np.empty(len(scores), dtype=np.float64)

    # Average ranks for tied values.
    i = 0

    while i < len(scores):

        j = i + 1

        while (
            j < len(scores)
            and sorted_scores[j] == sorted_scores[i]
        ):
            j += 1

        average_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = average_rank

        i = j

    positive_rank_sum = float(ranks[labels].sum())

    auc = (
        positive_rank_sum
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def top_fraction_metrics(scores, gt_mask, fraction):

    scores = np.asarray(scores, dtype=np.float64)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    flat = scores.ravel()
    gt_flat = gt_mask.ravel()

    n = len(flat)

    if n == 0:
        return {
            "selected_pixels": 0,
            "gt_precision": float("nan"),
            "gt_recall": float("nan"),
            "gt_overlap_dice": float("nan")
        }

    k = max(1, int(np.ceil(n * fraction)))

    # Select the highest-scoring pixels.
    indices = np.argpartition(flat, -k)[-k:]

    selected = np.zeros(n, dtype=bool)
    selected[indices] = True

    intersection = int(
        np.logical_and(selected, gt_flat).sum()
    )

    selected_count = int(selected.sum())
    gt_count = int(gt_flat.sum())

    precision = intersection / max(selected_count, 1)
    recall = intersection / max(gt_count, 1)

    dice = (
        2.0 * intersection
        / max(selected_count + gt_count, 1)
    )

    return {
        "selected_pixels": selected_count,
        "gt_precision": float(precision),
        "gt_recall": float(recall),
        "gt_overlap_dice": float(dice)
    }


# ---------------------------------------------------------
# SPATIAL METRICS
# ---------------------------------------------------------

def spatial_metrics(map_array, gt_mask, prefix):

    score_map = as_2d(map_array, prefix)

    if score_map.shape != gt_mask.shape:
        raise ValueError(
            f"{prefix} shape mismatch: "
            f"{score_map.shape} vs {gt_mask.shape}"
        )

    gt = np.asarray(gt_mask, dtype=bool)
    outside = ~gt

    inside_values = score_map[gt]
    outside_values = score_map[outside]

    result = {}

    result[f"{prefix}_mean_inside_gt"] = (
        float(inside_values.mean())
        if inside_values.size else float("nan")
    )

    result[f"{prefix}_mean_outside_gt"] = (
        float(outside_values.mean())
        if outside_values.size else float("nan")
    )

    result[f"{prefix}_median_inside_gt"] = (
        float(np.median(inside_values))
        if inside_values.size else float("nan")
    )

    result[f"{prefix}_median_outside_gt"] = (
        float(np.median(outside_values))
        if outside_values.size else float("nan")
    )

    inside_mass = float(inside_values.sum())
    total_mass = float(score_map.sum())

    result[f"{prefix}_inside_mass_fraction"] = (
        inside_mass / total_mass
        if abs(total_mass) > 1e-12 else float("nan")
    )

    result[f"{prefix}_rank_auc"] = rank_auc(
        score_map,
        gt
    )

    for fraction in (0.10, 0.20):

        metrics = top_fraction_metrics(
            score_map,
            gt,
            fraction
        )

        suffix = f"top{int(fraction * 100)}"

        for key, value in metrics.items():

            result[f"{prefix}_{suffix}_{key}"] = value

    return result


def signed_ig_metrics(ig_map, gt_mask):

    ig = as_2d(ig_map, "integrated_gradients")
    gt = np.asarray(gt_mask, dtype=bool)

    if ig.shape != gt.shape:
        raise ValueError(
            f"IG shape mismatch: {ig.shape} vs {gt.shape}"
        )

    positive = np.maximum(ig, 0.0)
    negative_magnitude = np.maximum(-ig, 0.0)
    absolute = np.abs(ig)

    result = {}

    result.update(
        spatial_metrics(
            positive,
            gt,
            "ig_positive"
        )
    )

    result.update(
        spatial_metrics(
            negative_magnitude,
            gt,
            "ig_negative_magnitude"
        )
    )

    result.update(
        spatial_metrics(
            absolute,
            gt,
            "ig_absolute"
        )
    )

    result["ig_signed_sum_inside_gt"] = float(
        ig[gt].sum()
    )

    result["ig_signed_mean_inside_gt"] = float(
        ig[gt].mean()
    ) if gt.any() else float("nan")

    result["ig_signed_sum_outside_gt"] = float(
        ig[~gt].sum()
    )

    return result


# ---------------------------------------------------------
# MAIN AUDIT
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        action="append",
        help="Case stem; can be repeated."
    )

    parser.add_argument(
        "--ig-steps",
        type=int,
        default=32
    )

    parser.add_argument(
        "--occlusion-window",
        type=int,
        default=32
    )

    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=16
    )

    args = parser.parse_args()

    if args.ig_steps < 1:
        parser.error("--ig-steps must be at least 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("SEGMENTATION XAI SPATIAL AUDIT")
    print("Device:", device)

    model, cfg, checkpoint_path = load_model(device)

    image_dir = (
        ROOT / "dataset" / "brisc2025" /
        "segmentation_task" / "test" / "images"
    )

    mask_dir = (
        ROOT / "dataset" / "brisc2025" /
        "segmentation_task" / "test" / "masks"
    )

    output_dir = (
        ROOT / "report" / "segmentation_xai_spatial"
    )

    raw_dir = output_dir / "raw_maps"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    xai = SegmentationXAI(model)

    rows = []

    selected_cases = args.case if args.case else CASES

    try:

        for stem in selected_cases:

            print(f"\nProcessing {stem}")

            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            # ---------------------------------------------
            # PREPROCESS IMAGE
            # ---------------------------------------------

            image = Image.open(image_path).convert("L")
            prep = core.prepare(image, cfg)

            x = torch.from_numpy(
                prep["xs"]
            ).to(device)

            # ---------------------------------------------
            # MODEL INFERENCE
            # ---------------------------------------------

            with torch.no_grad():

                logits = model(x).float()

                probability = torch.sigmoid(
                    logits
                )[0, 0].cpu().numpy()

            threshold = float(
                cfg.get("seg_thr", 0.5)
            )

            prediction = probability >= threshold

            # ---------------------------------------------
            # PREPROCESS GROUND TRUTH
            # ---------------------------------------------

            from src.preprocessing.cache import process_mask

            raw_mask = np.asarray(
                Image.open(mask_path).convert("L")
            )

            gt = process_mask(
                raw_mask,
                cfg.get(
                    "seg_resize_policy",
                    cfg["resize_policy"]
                ),
                cfg["img_size"]
            ) > 0

            if gt.shape != prediction.shape:
                raise RuntimeError(
                    f"Mask shape mismatch: {gt.shape} "
                    f"vs {prediction.shape}"
                )

            gt_pixels = int(gt.sum())
            pred_pixels = int(prediction.sum())

            intersection = int(
                np.logical_and(
                    gt,
                    prediction
                ).sum()
            )

            dice = (
                2.0 * intersection
                / max(gt_pixels + pred_pixels, 1)
            )

            # ---------------------------------------------
            # TARGETS
            # ---------------------------------------------

            targets = [
                ("predicted_region", prediction)
            ]

            if gt.any():
                targets.append(
                    ("ground_truth_oracle_OFFLINE", gt)
                )

            # ---------------------------------------------
            # BLACK BASELINE
            # ---------------------------------------------

            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=device
            ).view(1, 3, 1, 1)

            std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=device
            ).view(1, 3, 1, 1)

            black_baseline = (0.0 - mean) / std

            black_occlusion = tuple(
                float(v)
                for v in black_baseline.flatten().tolist()
            )

            # ---------------------------------------------
            # EXPLANATION TARGET LOOP
            # ---------------------------------------------

            for target_name, roi in targets:

                if not roi.any():

                    print(
                        f"  Skipping {target_name}: empty ROI"
                    )

                    rows.append({
                        "case": stem,
                        "target": target_name,
                        "status": "empty ROI; skipped",
                        "dice": dice,
                        "ground_truth_pixels": gt_pixels,
                        "prediction_pixels": pred_pixels
                    })

                    continue

                print(
                    f"  Explaining {target_name}"
                )

                # IMPORTANT:
                # Do not wrap this call in torch.no_grad().
                result = xai.explain(
                    x,
                    roi.astype(np.float32),
                    baseline=black_baseline,
                    occlusion_baseline=black_occlusion,
                    ig_steps=args.ig_steps,
                    occlusion_window=args.occlusion_window,
                    occlusion_stride=args.occlusion_stride
                )

                cam = as_2d(
                    result["gradcam"],
                    "gradcam"
                )

                ig = as_2d(
                    result["integrated_gradients"],
                    "integrated_gradients"
                )

                occ = as_2d(
                    result["occlusion"],
                    "occlusion"
                )

                if (
                    cam.shape != gt.shape
                    or ig.shape != gt.shape
                    or occ.shape != gt.shape
                ):
                    raise RuntimeError(
                        f"Explanation map shape mismatch for {stem}. "
                        f"GT={gt.shape}, CAM={cam.shape}, "
                        f"IG={ig.shape}, OCC={occ.shape}"
                    )

                # -----------------------------------------
                # SAVE RAW ARRAYS
                # -----------------------------------------

                safe_target = target_name.replace(
                    "/",
                    "_"
                )

                raw_path = (
                    raw_dir /
                    f"{stem}_{safe_target}.npz"
                )

                np.savez_compressed(
                    raw_path,
                    gradcam=cam.astype(np.float32),
                    integrated_gradients=ig.astype(np.float32),
                    occlusion=occ.astype(np.float32),
                    ground_truth=gt.astype(np.uint8),
                    prediction=prediction.astype(np.uint8),
                    probability=probability.astype(np.float32),
                    input_image=prep["arr_s"].astype(np.uint8),
                    target_name=np.asarray(target_name),
                    target_score=np.asarray(
                        float(result["target_score"])
                    )
                )

                # -----------------------------------------
                # CALCULATE SPATIAL METRICS
                # -----------------------------------------

                row = {
                    "case": stem,
                    "target": target_name,
                    "status": "ok",
                    "checkpoint": str(checkpoint_path),
                    "device": str(device),
                    "dice": float(dice),
                    "ground_truth_pixels": gt_pixels,
                    "prediction_pixels": pred_pixels,
                    "probability_max": float(probability.max()),
                    "target_score": float(
                        result["target_score"]
                    ),
                    "ig_attribution_sum": float(
                        result["ig_attribution_sum"]
                    ),
                    "raw_maps_file": str(raw_path)
                }

                # Grad-CAM: positive activation concentration.
                cam_positive = np.maximum(cam, 0.0)

                row.update(
                    spatial_metrics(
                        cam_positive,
                        gt,
                        "gradcam"
                    )
                )

                # Integrated Gradients: signed and absolute maps.
                row.update(
                    signed_ig_metrics(
                        ig,
                        gt
                    )
                )

                # Occlusion: preserve the signed raw map and
                # evaluate both positive and negative changes.
                row.update(
                    spatial_metrics(
                        np.maximum(occ, 0.0),
                        gt,
                        "occlusion_positive"
                    )
                )

                row.update(
                    spatial_metrics(
                        np.abs(occ),
                        gt,
                        "occlusion_absolute"
                    )
                )

                rows.append(row)

                print(
                    f"  Saved raw maps: {raw_path.name}"
                )

                print(
                    f"  Grad-CAM AUC: "
                    f"{row['gradcam_rank_auc']:.4f}"
                )

                print(
                    f"  IG absolute AUC: "
                    f"{row['ig_absolute_rank_auc']:.4f}"
                )

                print(
                    f"  Occlusion absolute AUC: "
                    f"{row['occlusion_absolute_rank_auc']:.4f}"
                )

            print(
                f"Completed {stem}: "
                f"GT={gt_pixels}, "
                f"pred={pred_pixels}, "
                f"Dice={dice:.5f}"
            )

    finally:
        xai.close()

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    csv_path = (
        output_dir /
        "segmentation_xai_spatial_metrics.csv"
    )

    if rows:

        fieldnames = sorted({
            key
            for row in rows
            for key in row
        })

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    print("\nSPATIAL AUDIT FINISHED")

    print("CSV:", csv_path)
    print("Raw maps:", raw_dir)

    print(
        "\nInterpretation warning: spatial overlap and rank AUC "
        "do not establish explanation faithfulness, model "
        "correctness, or clinical validity."
    )

    print(
        "Ground-truth-targeted explanations are offline-only."
    )


if __name__ == "__main__":
    main()
