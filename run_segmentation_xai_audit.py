
"""
Run segmentation XAI on selected BRISC2025 test failures.

Run from repository root:
    python run_segmentation_xai_audit.py

Run a single case:
    python run_segmentation_xai_audit.py --case brisc2025_test_00045_gl_ax_t1

Run all selected cases:
    python run_segmentation_xai_audit.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PROJECT PATHS
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
# LOAD EXISTING TRAINED SEGMENTATION MODEL
# ---------------------------------------------------------

def load_model(device):

    deploy_dir = ROOT / "models" / "deploy"
    cfg = core.load_cfg(deploy_dir)

    primary_config_path = (
        ROOT / "configs" / "segmentation_primary.json"
    )

    primary_config = json.loads(
        primary_config_path.read_text(encoding="utf-8")
    )

    primary = primary_config["primary"]

    checkpoint_path = (
        ROOT / "models" / f"{primary}_best.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Segmentation checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_config = checkpoint.get("cfg", {})

    model = UNetPlusPlus(
        pretrained=False,
        nested=model_config.get("nested", True)
    )

    if model_config.get("variant", "b0") == "b1":
        model.encoder = EffB1Encoder(pretrained=False)

    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    return model, cfg, checkpoint_path


# ---------------------------------------------------------
# NORMALIZE MAP FOR VISUALIZATION
# ---------------------------------------------------------

def norm_map(array):

    array = np.asarray(array, dtype=np.float32)

    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    minimum = float(array.min())
    maximum = float(array.max())

    return (array - minimum) / max(maximum - minimum, 1e-8)


# ---------------------------------------------------------
# MAIN AUDIT
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        action="append",
        help="Case stem. Can be supplied multiple times."
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

    if args.occlusion_window < 1:
        parser.error("--occlusion-window must be at least 1")

    if args.occlusion_stride < 1:
        parser.error("--occlusion-stride must be at least 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("SEGMENTATION XAI FAILURE AUDIT")
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
        ROOT / "report" / "segmentation_xai"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    xai = SegmentationXAI(model)

    rows = []

    selected_cases = args.case if args.case else CASES

    # -----------------------------------------------------
    # PROCESS CASES
    # -----------------------------------------------------

    try:

        for stem in selected_cases:

            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            print(f"\nProcessing: {stem}")

            # ---------------------------------------------
            # IMAGE PREPROCESSING
            # ---------------------------------------------

            image = Image.open(image_path).convert("L")

            prep = core.prepare(image, cfg)

            x = torch.from_numpy(
                prep["xs"]
            ).to(device)

            # ---------------------------------------------
            # SEGMENTATION INFERENCE
            # ---------------------------------------------

            # no_grad is restricted to inference only.
            # Grad-CAM must execute outside this context.
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
            # GROUND-TRUTH MASK PREPROCESSING
            # ---------------------------------------------

            raw_mask = np.asarray(
                Image.open(mask_path).convert("L")
            )

            from src.preprocessing.cache import process_mask

            gt_mask = process_mask(
                raw_mask,
                cfg.get(
                    "seg_resize_policy",
                    cfg["resize_policy"]
                ),
                cfg["img_size"]
            ) > 0

            if gt_mask.shape != prediction.shape:

                raise RuntimeError(
                    f"GT/prediction shape mismatch for {stem}: "
                    f"{gt_mask.shape} vs {prediction.shape}"
                )

            gt_pixels = int(gt_mask.sum())
            pred_pixels = int(prediction.sum())

            intersection = int(
                np.logical_and(
                    prediction,
                    gt_mask
                ).sum()
            )

            dice = (
                2.0 * intersection /
                max(pred_pixels + gt_pixels, 1)
            )

            # ---------------------------------------------
            # DEFINE EXPLANATION TARGETS
            # ---------------------------------------------

            target_specs = [
                ("predicted_region", prediction)
            ]

            if gt_mask.any():

                target_specs.append(
                    ("ground_truth_oracle_OFFLINE", gt_mask)
                )

            # ---------------------------------------------
            # NORMALIZED BLACK BASELINE
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
                float(value)
                for value in black_baseline.flatten().tolist()
            )

            # ---------------------------------------------
            # GENERATE EXPLANATIONS
            # ---------------------------------------------

            for target_name, roi in target_specs:

                if not roi.any():

                    rows.append({
                        "case": stem,
                        "target": target_name,
                        "status": "empty ROI; explanation skipped",
                        "prediction_pixels": pred_pixels,
                        "ground_truth_pixels": gt_pixels,
                        "dice": dice,
                        "checkpoint": str(checkpoint_path),
                        "device": str(device)
                    })

                    print(
                        f"  Skipped {target_name}: empty ROI"
                    )

                    continue

                print(
                    f"  Generating explanation: {target_name}"
                )

                # IMPORTANT:
                # This call must remain outside torch.no_grad().
                result = xai.explain(
                    x,
                    roi.astype(np.float32),
                    baseline=black_baseline,
                    occlusion_baseline=black_occlusion,
                    ig_steps=args.ig_steps,
                    occlusion_window=args.occlusion_window,
                    occlusion_stride=args.occlusion_stride
                )

                # -----------------------------------------
                # CREATE DIAGNOSTIC FIGURE
                # -----------------------------------------

                fig, axes = plt.subplots(
                    1,
                    6,
                    figsize=(22, 4)
                )

                axes[0].imshow(
                    prep["arr_s"],
                    cmap="gray"
                )
                axes[0].set_title("MRI")

                axes[1].imshow(
                    gt_mask,
                    cmap="gray"
                )
                axes[1].set_title("Ground truth")

                axes[2].imshow(
                    prediction,
                    cmap="gray"
                )
                axes[2].set_title("Prediction")

                axes[3].imshow(
                    norm_map(result["gradcam"]),
                    cmap="jet",
                    vmin=0,
                    vmax=1
                )
                axes[3].set_title(
                    "Segmentation Grad-CAM"
                )

                ig_map = np.asarray(
                    result["integrated_gradients"]
                )

                ig_limit = max(
                    float(np.max(np.abs(ig_map))),
                    1e-8
                )

                axes[4].imshow(
                    ig_map,
                    cmap="seismic",
                    vmin=-ig_limit,
                    vmax=ig_limit
                )
                axes[4].set_title(
                    "Integrated Gradients (signed)"
                )

                axes[5].imshow(
                    norm_map(result["occlusion"]),
                    cmap="magma",
                    vmin=0,
                    vmax=1
                )
                axes[5].set_title(
                    "Occlusion sensitivity"
                )

                for axis in axes:
                    axis.axis("off")

                target_score = float(
                    result["target_score"]
                )

                fig.suptitle(
                    f"{stem} | target={target_name} "
                    f"| score={target_score:.5g}"
                )

                fig.tight_layout()

                safe_target = target_name.replace(
                    "/",
                    "_"
                )

                figure_path = (
                    output_dir /
                    f"{stem}_{safe_target}.png"
                )

                fig.savefig(
                    figure_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close(fig)

                # -----------------------------------------
                # SAVE AUDIT RESULTS
                # -----------------------------------------

                rows.append({
                    "case": stem,
                    "target": target_name,
                    "status": "ok",
                    "target_score": target_score,
                    "ig_attribution_sum": float(
                        result["ig_attribution_sum"]
                    ),
                    "prediction_pixels": pred_pixels,
                    "ground_truth_pixels": gt_pixels,
                    "dice": dice,
                    "checkpoint": str(checkpoint_path),
                    "device": str(device),
                    "figure": str(figure_path)
                })

                print(
                    f"  Saved: {figure_path.name}"
                )

            print(
                f"Completed {stem}: "
                f"GT={gt_pixels}, "
                f"pred={pred_pixels}, "
                f"pmax={float(probability.max()):.5f}, "
                f"Dice={dice:.5f}"
            )

    finally:

        # Close hooks and release XAI resources.
        xai.close()

    # -----------------------------------------------------
    # SAVE CSV AFTER ALL CASES
    # -----------------------------------------------------

    csv_path = (
        output_dir / "segmentation_xai_audit.csv"
    )

    if rows:

        fieldnames = sorted({
            key
            for row in rows
            for key in row.keys()
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

    print("\nSEGMENTATION XAI AUDIT FINISHED")

    print(
        "Saved figures and audit CSV under:",
        output_dir
    )

    print(
        "Ground-truth-targeted explanations are offline "
        "diagnostic analyses only."
    )


if __name__ == "__main__":
    main()
