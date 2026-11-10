"""Read-only ONNX vs PyTorch parity check for one MRI image.
Save this file in the project ROOT (folder containing api/, src/, configs/, models/).
Run from VS Code; a file picker will let you select the MRI image.
"""
from pathlib import Path
import json, sys, traceback
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")


def choose_image():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    chosen = filedialog.askopenfilename(
        title="Select one MRI image for ONNX/PyTorch parity test",
        filetypes=[("MRI/image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(chosen) if chosen else None


def compare_arrays(label, pt, ox):
    pt = np.asarray(pt, dtype=np.float32); ox = np.asarray(ox, dtype=np.float32)
    print(f"\n{label}")
    print(f"  PyTorch shape: {pt.shape} | ONNX shape: {ox.shape}")
    if pt.shape != ox.shape:
        print("  SHAPE MISMATCH — cannot compare."); return False
    if not np.isfinite(pt).all() or not np.isfinite(ox).all():
        print("  NaN/Inf detected."); return False
    d = np.abs(pt - ox)
    rel = np.linalg.norm((pt - ox).ravel()) / max(np.linalg.norm(pt.ravel()), 1e-12)
    print(f"  Max absolute difference:  {d.max():.8g}")
    print(f"  Mean absolute difference: {d.mean():.8g}")
    print(f"  Relative L2 difference:   {rel:.8g}")
    ok = bool(np.allclose(pt, ox, rtol=1e-3, atol=1e-4))
    print(f"  allclose (rtol=1e-3, atol=1e-4): {ok}")
    return ok


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    required = ["api/core.py", "src/classification/model.py", "src/segmentation/model.py",
                "models/deploy/deploy_config.json", "configs/segmentation_primary.json",
                "models/deploy/classifier.onnx", "models/deploy/segmenter.onnx",
                "models/deploy/classifier_best.pt"]
    missing = [x for x in required if not (ROOT / x).exists()]
    if missing:
        raise FileNotFoundError("Save this script in the project ROOT. Missing:\n" + "\n".join(missing))
    image_path = choose_image()
    if image_path is None:
        print("No image selected; cancelled."); return
    sys.path.insert(0, str(ROOT))
    from api import core
    from src.classification.model import build_efficientnet_b0
    from src.segmentation.model import UNetPlusPlus, EffB1Encoder

    deploy = ROOT / "models" / "deploy"
    cfg = core.load_cfg(deploy)
    seg_info = json.loads((ROOT / "configs" / "segmentation_primary.json").read_text(encoding="utf-8"))
    seg_name = seg_info["primary"]
    cls_path = deploy / "classifier_best.pt"
    seg_path = ROOT / "models" / f"{seg_name}_best.pt"
    if not seg_path.exists(): raise FileNotFoundError(f"Selected segmenter checkpoint missing: {seg_path}")

    print("=" * 70)
    print("BRAIN TUMOR PROJECT — ONNX / PYTORCH PARITY CHECK")
    print("=" * 70)
    print("Image:", image_path)
    print("Classifier checkpoint:", cls_path)
    print("Segmenter checkpoint:", seg_path, f"({seg_name})")
    print("Device: CPU")

    # Exact production preprocessing from api/core.py
    with Image.open(image_path) as im:
        prepared = core.prepare(im.convert("RGB"), cfg)
    xc = torch.from_numpy(prepared["xc"]).to(DEVICE, dtype=torch.float32)
    xs = torch.from_numpy(prepared["xs"]).to(DEVICE, dtype=torch.float32)

    cls_ck = torch.load(cls_path, map_location=DEVICE, weights_only=False)
    classifier = build_efficientnet_b0(pretrained=False).to(DEVICE)
    classifier.load_state_dict(cls_ck["model"], strict=True); classifier.eval()

    seg_ck = torch.load(seg_path, map_location=DEVICE, weights_only=False)
    scfg = seg_ck.get("cfg", {})
    segmenter = UNetPlusPlus(pretrained=False, nested=scfg.get("nested", True))
    if scfg.get("variant", "b0") == "b1": segmenter.encoder = EffB1Encoder(pretrained=False)
    segmenter = segmenter.to(DEVICE)
    segmenter.load_state_dict(seg_ck["model"], strict=True); segmenter.eval()

    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    cls_sess = ort.InferenceSession(str(deploy / "classifier.onnx"), providers=["CPUExecutionProvider"], sess_options=opts)
    seg_sess = ort.InferenceSession(str(deploy / "segmenter.onnx"), providers=["CPUExecutionProvider"], sess_options=opts)
    with torch.inference_mode():
        pt_logits = classifier(xc).cpu().numpy()
        pt_seg_logits = segmenter(xs).cpu().numpy()
    ox_logits = cls_sess.run(None, {cls_sess.get_inputs()[0].name: xc.cpu().numpy()})[0]
    ox_seg_logits = seg_sess.run(None, {seg_sess.get_inputs()[0].name: xs.cpu().numpy()})[0]

    print("\nONNX Runtime:", ort.__version__)
    print("Classifier providers:", cls_sess.get_providers())
    print("Segmenter providers:", seg_sess.get_providers())
    cls_ok = compare_arrays("CLASSIFIER LOGITS", pt_logits, ox_logits)
    seg_ok = compare_arrays("SEGMENTATION LOGITS", pt_seg_logits, ox_seg_logits)

    T = float(cfg["temperature"]); classes = cfg["classes"]
    p_pt = softmax(pt_logits / T)[0]; p_ox = softmax(ox_logits / T)[0]
    print("\nTEMPERATURE-SCALED CLASS PROBABILITIES")
    for name, a, b in zip(classes, p_pt, p_ox):
        print(f"  {name:12s} PyTorch={a:.8f} ONNX={b:.8f} abs_diff={abs(a-b):.8g}")
    print("PyTorch predicted class:", classes[int(p_pt.argmax())])
    print("ONNX predicted class:   ", classes[int(p_ox.argmax())])
    print("Probability max abs difference:", float(np.max(np.abs(p_pt-p_ox))))

    mask_agreement = None
    if pt_seg_logits.shape == ox_seg_logits.shape:
        m_pt = (1 / (1 + np.exp(-pt_seg_logits))) > float(cfg["seg_thr"])
        m_ox = (1 / (1 + np.exp(-ox_seg_logits))) > float(cfg["seg_thr"])
        mask_agreement = float(np.mean(m_pt == m_ox) * 100)
        print("\nSEGMENTATION MASKS (threshold", cfg["seg_thr"], ")")
        print("Pixel agreement:", f"{mask_agreement:.6f}%")
        print("PyTorch positive pixels:", int(m_pt.sum()))
        print("ONNX positive pixels:   ", int(m_ox.sum()))

    report = ROOT / "onnx_pytorch_parity_report.txt"
    report.write_text(
        "ONNX/PyTorch parity check\n"
        f"Image: {image_path}\nClassifier logits allclose: {cls_ok}\n"
        f"Segmenter logits allclose: {seg_ok}\nTemperature: {T}\n"
        f"PyTorch class: {classes[int(p_pt.argmax())]}\nONNX class: {classes[int(p_ox.argmax())]}\n"
        f"Classifier probability max abs difference: {float(np.max(np.abs(p_pt-p_ox)))}\n"
        f"Segmentation mask pixel agreement: {mask_agreement}\n"
        "See VS Code terminal for full numerical differences.\n", encoding="utf-8")
    print("\n" + "=" * 70)
    print("RESULT:", "PASS under stated tolerance" if cls_ok and seg_ok else "NEEDS INVESTIGATION")
    print("One-image implementation check only; this is not an accuracy evaluation.")
    print("Report saved:", report)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("Parity test error", "An error occurred. See the VS Code terminal for details.")
            r.destroy()
        except Exception:
            pass
