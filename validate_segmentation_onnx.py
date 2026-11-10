"""
BRISC2025 segmentation validation: PyTorch vs ONNX + ground-truth metrics.

Save this file in the PROJECT ROOT (the folder containing api/, src/, configs/, models/).
Run it in VS Code using "Run Python File". It opens file pickers for an MRI image
and, if it cannot auto-match it, the corresponding ground-truth mask.

Read-only: does not modify model weights, ONNX files, datasets, or application code.
"""
from pathlib import Path
import json
import sys
import traceback

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pick_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    name = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return Path(name) if name else None


def find_matching_mask(image_path):
    """Look for same-stem mask in the BRISC segmentation test mask folder."""
    mask_dir = ROOT / "dataset" / "brisc2025" / "segmentation_task" / "test" / "masks"
    if mask_dir.is_dir():
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            candidate = mask_dir / (image_path.stem + ext)
            if candidate.is_file():
                return candidate
    return None


def load_mask_as_training_cache(mask_path, size, policy):
    """Mirror src.preprocessing.cache.process_mask exactly."""
    from src.preprocessing.cache import process_mask
    with Image.open(mask_path) as im:
        gray = np.asarray(im.convert("L"))
    return process_mask(gray, policy, size).astype(bool)


def binary_metrics(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "pixel_accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_pixels": int(pred.sum()), "truth_pixels": int(truth.sum())
    }


def compare_arrays(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    print(f"\n{name}")
    print(f"  PyTorch shape: {a.shape} | ONNX shape: {b.shape}")
    if a.shape != b.shape:
        print("  SHAPE MISMATCH — cannot compare.")
        return False
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        print("  NaN/Inf detected.")
        return False
    d = np.abs(a - b)
    rel = float(np.linalg.norm((a - b).ravel()) /
                max(float(np.linalg.norm(a.ravel())), 1e-12))
    close = bool(np.allclose(a, b, rtol=1e-3, atol=1e-4))
    print(f"  Max abs difference:  {float(d.max()):.8g}")
    print(f"  Mean abs difference: {float(d.mean()):.8g}")
    print(f"  Relative L2 diff:    {rel:.8g}")
    print(f"  allclose (1e-3, 1e-4): {close}")
    return close


def print_metrics(label, m):
    print(f"\n{label}")
    print(f"  Dice:          {m['dice']:.6f}")
    print(f"  IoU:           {m['iou']:.6f}")
    print(f"  Precision:     {m['precision']:.6f}")
    print(f"  Recall:        {m['recall']:.6f}")
    print(f"  Pixel accuracy:{m['pixel_accuracy']:.6f}")
    print(f"  Predicted fg pixels: {m['pred_pixels']}")
    print(f"  Ground-truth fg pixels: {m['truth_pixels']}")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


def main():
    required = [
        ROOT / "api" / "core.py",
        ROOT / "src" / "segmentation" / "model.py",
        ROOT / "src" / "preprocessing" / "cache.py",
        ROOT / "configs" / "segmentation_primary.json",
        ROOT / "models" / "deploy" / "deploy_config.json",
        ROOT / "models" / "deploy" / "segmenter.onnx",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Save this script in the project ROOT. Missing required paths:\n" +
            "\n".join(str(p) for p in missing)
        )

    image_path = pick_file(
        "Select a BRISC2025 segmentation TEST MRI image",
        [("MRI images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
         ("All files", "*.*")]
    )
    if not image_path:
        print("Cancelled: no image selected.")
        return

    mask_path = find_matching_mask(image_path)
    if mask_path is None:
        mask_path = pick_file(
            "Select the ground-truth mask corresponding to the selected MRI image",
            [("Mask images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
             ("All files", "*.*")]
        )
    if not mask_path:
        print("Cancelled: no ground-truth mask selected.")
        return

    sys.path.insert(0, str(ROOT))
    from api import core
    from src.preprocessing.cache import process_mask
    from src.segmentation.model import UNetPlusPlus, EffB1Encoder

    primary_cfg = json.loads(
        (ROOT / "configs" / "segmentation_primary.json").read_text(encoding="utf-8")
    )
    seg_tag = primary_cfg["primary"]
    checkpoint_path = ROOT / "models" / f"{seg_tag}_best.pt"
    onnx_path = ROOT / "models" / "deploy" / "segmenter.onnx"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint missing: {checkpoint_path}")

    deploy_dir = ROOT / "models" / "deploy"
    deploy_cfg = core.load_cfg(deploy_dir)
    size = int(deploy_cfg["img_size"])
    seg_policy = deploy_cfg.get("seg_resize_policy", deploy_cfg["resize_policy"])
    threshold = float(deploy_cfg["seg_thr"])

    print("=" * 72)
    print("BRISC2025 SEGMENTATION VALIDATION")
    print("=" * 72)
    print(f"MRI image: {image_path}")
    print(f"Ground-truth mask: {mask_path}")
    print(f"Segmentation checkpoint: {checkpoint_path}")
    print(f"ONNX model: {onnx_path}")
    print(f"Preprocessing policy: {seg_policy}; image size: {size}")
    print(f"Segmentation threshold: {threshold}")
    print("Execution device: CPU")

    # Exact production image preprocessing path used by the application.
    with Image.open(image_path) as im:
        prepared = core.prepare(im.convert("RGB"), deploy_cfg)
    xs = torch.from_numpy(prepared["xs"]).to(DEVICE, dtype=torch.float32)

    # Exact mask preprocessing used to create the training/evaluation cache.
    truth = load_mask_as_training_cache(mask_path, size, seg_policy)
    if truth.shape != (size, size):
        raise ValueError(f"Processed mask has unexpected shape: {truth.shape}")

    # Reconstruct the selected trained PyTorch segmenter.
    ck = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_cfg = ck.get("cfg", {})
    model = UNetPlusPlus(pretrained=False, nested=model_cfg.get("nested", True))
    if model_cfg.get("variant", "b0") == "b1":
        model.encoder = EffB1Encoder(pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    # ONNX Runtime session, explicitly CPU.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"], sess_options=options
    )
    input_name = session.get_inputs()[0].name

    with torch.inference_mode():
        pt_logits = model(xs).cpu().numpy()
    onnx_logits = session.run(None, {input_name: xs.cpu().numpy()})[0]

    parity = compare_arrays("SEGMENTATION LOGITS: PYTORCH vs ONNX", pt_logits, onnx_logits)
    if pt_logits.shape != onnx_logits.shape or pt_logits.ndim != 4:
        raise ValueError(
            f"Unexpected segmentation output shapes: PyTorch={pt_logits.shape}, ONNX={onnx_logits.shape}"
        )

    # Sigmoid then apply the deployment threshold, matching api/pipeline.py.
    pt_prob = 1.0 / (1.0 + np.exp(-pt_logits[0, 0].astype(np.float64)))
    ox_prob = 1.0 / (1.0 + np.exp(-onnx_logits[0, 0].astype(np.float64)))
    pt_mask = pt_prob > threshold
    ox_mask = ox_prob > threshold
    mask_agreement = float(np.mean(pt_mask == ox_mask))

    print("\nBINARY MASK AGREEMENT")
    print(f"  PyTorch vs ONNX pixel agreement: {100 * mask_agreement:.6f}%")
    print(f"  PyTorch positive pixels: {int(pt_mask.sum())}")
    print(f"  ONNX positive pixels:    {int(ox_mask.sum())}")
    print(f"  Ground-truth positive pixels: {int(truth.sum())}")
    print(f"  Mean probability abs difference: {float(np.mean(np.abs(pt_prob-ox_prob))):.8g}")
    print(f"  Max probability abs difference:  {float(np.max(np.abs(pt_prob-ox_prob))):.8g}")

    pt_metrics = binary_metrics(pt_mask, truth)
    ox_metrics = binary_metrics(ox_mask, truth)
    print_metrics("PYTORCH vs GROUND TRUTH", pt_metrics)
    print_metrics("ONNX vs GROUND TRUTH", ox_metrics)

    # Save a visual diagnostic: original image, GT, PyTorch mask, ONNX mask.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Image.open(image_path) as im:
        original = np.asarray(im.convert("L"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original, cmap="gray"); axes[0].set_title("MRI image")
    axes[1].imshow(truth, cmap="gray"); axes[1].set_title("Ground truth")
    axes[2].imshow(pt_mask, cmap="gray"); axes[2].set_title("PyTorch prediction")
    axes[3].imshow(ox_mask, cmap="gray"); axes[3].set_title("ONNX prediction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    visual_path = ROOT / "segmentation_parity_visual.png"
    fig.savefig(visual_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = ROOT / "segmentation_validation_report.txt"
    report = [
        "BRISC2025 SEGMENTATION VALIDATION",
        f"Image: {image_path}",
        f"Ground truth: {mask_path}",
        f"Checkpoint: {checkpoint_path}",
        f"ONNX: {onnx_path}",
        f"Threshold: {threshold}",
        f"Logits allclose (rtol=1e-3, atol=1e-4): {parity}",
        f"Binary mask pixel agreement: {mask_agreement:.8f}",
        f"PyTorch metrics: {pt_metrics}",
        f"ONNX metrics: {ox_metrics}",
        f"Visual diagnostic: {visual_path}",
        "",
        "One-image validation only; do not interpret as test-set performance."
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 72)
    print("TEST COMPLETE")
    print(f"Text report: {report_path}")
    print(f"Visual comparison: {visual_path}")
    print("This is a one-image check, not a full-dataset performance estimate.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nVALIDATION ERROR:\n")
        traceback.print_exc()
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Segmentation validation error",
                "The test stopped with an error. See the VS Code terminal for details."
            )
            root.destroy()
        except Exception:
            pass
