"""Full BRISC2025 segmentation test-set evaluation (read-only).

Save in the project root (the folder containing api/, src/, configs/, models/, dataset/).
Evaluates every paired test MRI/mask with both PyTorch and ONNX, using the project's
own preprocessing and deployment threshold. Writes all evaluation outputs under the existing singular report/ folder:
report/segmentation_full_testset/ and report/segmentation_error_analysis/.
It does not move, merge, rename, or delete report/ or reports/ contents.

This script does NOT train, tune thresholds, alter checkpoints, or modify the app.
It does not access metrics_report/ or reports/.
"""
from pathlib import Path
import json, re, sys, time, traceback
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "dataset" / "brisc2025" / "segmentation_task" / "test" / "images"
MASK_DIR = ROOT / "dataset" / "brisc2025" / "segmentation_task" / "test" / "masks"
OUT_DIR = ROOT / "report" / "segmentation_full_testset"
ERROR_DIR = ROOT / "report" / "segmentation_error_analysis"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 2026


def find_mask(image_path):
    for ext in EXTS:
        p = MASK_DIR / (image_path.stem + ext)
        if p.is_file():
            return p
    return None


def class_from_path(path):
    # BRISC filename tokens: gl=glioma, me=meningioma, pi=pituitary.
    # Healthy samples may have no tumour token; retain an explicit unknown label
    # rather than silently assigning a class.
    for token, label in (("gl", "glioma"), ("me", "meningioma"), ("pi", "pituitary")):
        if re.search(rf"_{token}_", path.stem.lower()):
            return label
    for part in path.parts:
        p = part.lower().replace("-", "_").replace(" ", "_")
        if p in {"glioma", "meningioma", "pituitary", "no_tumor", "healthy"}:
            return "no_tumor" if p == "healthy" else p
    return "unknown"


def metrics(pred, truth):
    pred = np.asarray(pred, bool); truth = np.asarray(truth, bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    # Explicitly define empty-GT cases: empty prediction is a perfect empty-mask
    # match; any predicted foreground is a false positive.
    denom_d = 2 * tp + fp + fn
    denom_i = tp + fp + fn
    dice = (2 * tp / denom_d) if denom_d else 1.0
    iou = (tp / denom_i) if denom_i else 1.0
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not truth.any() else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(dice=dice, iou=iou, precision=precision, recall=recall,
                specificity=specificity, pixel_accuracy=(tp + tn) / max(tp+tn+fp+fn, 1),
                tp=tp, fp=fp, fn=fn, tn=tn, pred_pixels=int(pred.sum()),
                gt_pixels=int(truth.sum()), gt_empty=int(not truth.any()),
                pred_empty=int(not pred.any()))


def bootstrap_ci(values, reps=BOOTSTRAP_REPS, seed=RANDOM_SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0: return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]



def orientation_from_name(name):
    stem = Path(name).stem.lower()
    for token, label in (("ax", "axial"), ("co", "coronal"), ("sa", "sagittal")):
        if re.search(rf"_{token}_", stem):
            return label
    return "unknown"


def make_error_overlay(gray, pred, truth):
    """Gray background; TP=green, FP=red, FN=blue."""
    base = np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)
    out = (base.astype(np.float32) * 0.48).astype(np.uint8)
    out[pred & truth] = (35, 190, 85)
    out[pred & ~truth] = (235, 65, 65)
    out[~pred & truth] = (55, 125, 245)
    return out


def create_error_analysis(good, core, process_mask, cfg, session, in_name, policy, size, threshold):
    """Visualize the 20 lowest-Dice cases and summarize filename-derived subgroups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    worst = good.nsmallest(min(20, len(good)), "onnx_dice")
    rows = []
    contact = []
    for rank, (_, row) in enumerate(worst.iterrows(), 1):
        image_name, mask_name = str(row["image"]), str(row["mask"])
        image_path, mask_path = IMAGE_DIR / image_name, MASK_DIR / mask_name
        with Image.open(image_path) as im:
            prepared = core.prepare(im.convert("RGB"), cfg)
        with Image.open(mask_path) as mi:
            truth = process_mask(np.asarray(mi.convert("L")), policy, size).astype(bool)
        x = np.asarray(prepared["xs"], dtype=np.float32)
        if x.ndim == 3:
            x = x[None, ...]
        logits = session.run(None, {in_name: x})[0]
        z = logits[0, 0].astype(np.float64)
        # Stable sigmoid avoids overflow for extreme logits.
        prob = np.empty_like(z)
        positive = z >= 0
        prob[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        ez = np.exp(z[~positive])
        prob[~positive] = ez / (1.0 + ez)
        pred = prob > threshold
        gray = np.asarray(prepared["arr_s"], dtype=np.uint8)
        if truth.shape != pred.shape:
            raise ValueError(f"GT/prediction shape mismatch for {image_name}: {truth.shape} vs {pred.shape}")

        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        tn = int(np.logical_and(~pred, ~truth).sum())
        denom = 2 * tp + fp + fn
        dice = (2 * tp / denom) if denom else 1.0
        iou_denom = tp + fp + fn
        iou = (tp / iou_denom) if iou_denom else 1.0
        rows.append({
            "rank": rank, "image": image_name, "mask": mask_name,
            "class_from_filename": class_from_path(Path(image_name)),
            "orientation_from_filename": orientation_from_name(image_name),
            "dice": dice, "iou": iou, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "gt_pixels": int(truth.sum()), "pred_pixels": int(pred.sum()),
            "threshold": threshold,
        })
        overlay = make_error_overlay(gray, pred, truth)
        fig, axes = plt.subplots(1, 5, figsize=(17, 4))
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("MRI (preprocessed)")
        axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"Ground truth\n{int(truth.sum())} px")
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title(f"ONNX prediction\n{int(pred.sum())} px")
        axes[3].imshow(overlay); axes[3].set_title("Error map: TP / FP / FN")
        axes[4].imshow(prob, cmap="magma", vmin=0, vmax=1); axes[4].set_title("Probability map")
        for ax in axes: ax.axis("off")
        fig.suptitle(f"{image_name} | Dice={dice:.4f} | FP={fp} | FN={fn}", fontsize=10)
        fig.tight_layout()
        fig.savefig(ERROR_DIR / f"case_{rank:02d}_{Path(image_name).stem}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        contact.append((image_name, overlay, dice))

    failure_df = pd.DataFrame(rows)
    failure_df.to_csv(ERROR_DIR / "failure_case_metrics.csv", index=False)
    cols = 4
    nrows = int(np.ceil(len(contact) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, overlay, dice) in zip(axes, contact):
        ax.imshow(overlay); ax.set_title(f"{name}\nDice={dice:.3f}", fontsize=8); ax.axis("off")
    for ax in axes[len(contact):]: ax.axis("off")
    fig.suptitle("BRISC2025 worst-case segmentation errors (green=TP, red=FP, blue=FN)")
    fig.tight_layout()
    fig.savefig(ERROR_DIR / "worst_cases_contact_sheet.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Descriptive summaries only; class/orientation labels are filename-derived.
    all_df = good.copy()
    all_df["orientation_from_filename"] = all_df["image"].map(orientation_from_name)
    all_df.groupby("class", dropna=False)["onnx_dice"].agg(
        count="count", mean="mean", median="median", std="std"
    ).reset_index().to_csv(ERROR_DIR / "full_testset_by_filename_class.csv", index=False)
    all_df.groupby("orientation_from_filename", dropna=False)["onnx_dice"].agg(
        count="count", mean="mean", median="median", std="std"
    ).reset_index().to_csv(ERROR_DIR / "full_testset_by_orientation.csv", index=False)
    print(f"\nError analysis: visualized {len(rows)} worst cases in {ERROR_DIR}")


def main():
    required = [IMAGE_DIR, MASK_DIR, ROOT/"api"/"core.py",
                ROOT/"src"/"preprocessing"/"cache.py",
                ROOT/"src"/"segmentation"/"model.py",
                ROOT/"configs"/"segmentation_primary.json",
                ROOT/"models"/"deploy"/"deploy_config.json",
                ROOT/"models"/"deploy"/"segmenter.onnx"]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing project paths:\n" + "\n".join(missing))
    sys.path.insert(0, str(ROOT))
    from api import core
    from src.preprocessing.cache import process_mask
    from src.segmentation.model import UNetPlusPlus, EffB1Encoder

    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    pairs = [(p, find_mask(p)) for p in images]
    missing_masks = [str(p) for p, m in pairs if m is None]
    pairs = [(p, m) for p, m in pairs if m is not None]
    if not pairs: raise RuntimeError("No image/mask pairs found. Check matching filenames.")

    primary = json.loads((ROOT/"configs"/"segmentation_primary.json").read_text(encoding="utf-8"))
    tag = primary["primary"]
    ckpt_path = ROOT/"models"/f"{tag}_best.pt"
    if not ckpt_path.is_file(): raise FileNotFoundError(ckpt_path)
    deploy_dir = ROOT/"models"/"deploy"
    cfg = core.load_cfg(deploy_dir)
    size = int(cfg["img_size"])
    policy = cfg.get("seg_resize_policy", cfg["resize_policy"])
    threshold = float(cfg["seg_thr"])

    print("="*78)
    print("BRISC2025 FULL SEGMENTATION TEST-SET EVALUATION")
    print(f"Found MRI images: {len(images)} | matched pairs: {len(pairs)} | missing masks: {len(missing_masks)}")
    print(f"Checkpoint: {ckpt_path.name} | ONNX: models/deploy/segmenter.onnx")
    print(f"Preprocessing: {policy}, {size}x{size} | threshold: {threshold}")
    print("Running PyTorch and ONNX on CPU; no training or threshold tuning.")

    device = torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ck.get("cfg", {})
    model = UNetPlusPlus(pretrained=False, nested=mc.get("nested", True))
    if mc.get("variant", "b0") == "b1": model.encoder = EffB1Encoder(pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ROOT/"models"/"deploy"/"segmenter.onnx"),
                                sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    records=[]; parity_abs_max=[]; parity_abs_mean=[]; parity_rel=[]; parity_agree=[]
    t0=time.time()
    for idx, (img_path, mask_path) in enumerate(pairs, 1):
        try:
            with Image.open(img_path) as im:
                prepared = core.prepare(im.convert("RGB"), cfg)
            x = np.asarray(prepared["xs"], dtype=np.float32)
            if x.ndim == 3: x = x[None, ...]
            if x.shape != (1, 3, size, size):
                raise ValueError(f"Unexpected model input shape {x.shape}")
            with Image.open(mask_path) as mi: mask_gray = np.asarray(mi.convert("L"))
            gt = process_mask(mask_gray, policy, size).astype(bool)
            xt = torch.from_numpy(x).to(device)
            with torch.inference_mode(): pt_logits = model(xt).cpu().numpy()
            on_logits = sess.run(None, {in_name: x})[0]
            if pt_logits.shape != on_logits.shape or pt_logits.shape != (1,1,size,size):
                raise ValueError(f"Output shape mismatch: PT {pt_logits.shape}, ONNX {on_logits.shape}")
            pt_prob = 1.0/(1.0+np.exp(-pt_logits[0,0].astype(np.float64)))
            on_prob = 1.0/(1.0+np.exp(-on_logits[0,0].astype(np.float64)))
            pt_mask = pt_prob > threshold; on_mask = on_prob > threshold
            pm = metrics(pt_mask, gt); om = metrics(on_mask, gt)
            delta = np.abs(pt_logits.astype(np.float64)-on_logits.astype(np.float64))
            rel = float(np.linalg.norm((pt_logits-on_logits).ravel()) /
                        max(float(np.linalg.norm(pt_logits.ravel())), 1e-12))
            agree = float(np.mean(pt_mask == on_mask))
            parity_abs_max.append(float(delta.max())); parity_abs_mean.append(float(delta.mean()))
            parity_rel.append(rel); parity_agree.append(agree)
            cls = class_from_path(img_path)
            rec = {"image":img_path.name, "mask":mask_path.name, "class":cls,
                   **{f"pt_{k}":v for k,v in pm.items()},
                   **{f"onnx_{k}":v for k,v in om.items()},
                   "pt_onnx_mask_agreement":agree,
                   "logit_max_abs_diff":float(delta.max()),
                   "logit_mean_abs_diff":float(delta.mean()),
                   "logit_relative_l2_diff":rel,
                   "prob_mean_abs_diff":float(np.mean(np.abs(pt_prob-on_prob))),
                   "prob_max_abs_diff":float(np.max(np.abs(pt_prob-on_prob)))}
            records.append(rec)
            if idx % 25 == 0 or idx == len(pairs):
                print(f"Processed {idx:4d}/{len(pairs)} | elapsed {time.time()-t0:.1f}s")
        except Exception as e:
            records.append({"image":img_path.name,"mask":mask_path.name,
                            "class":class_from_path(img_path),"error":repr(e)})
            print(f"ERROR [{idx}/{len(pairs)}] {img_path.name}: {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(records)
    csv_path=OUT_DIR/"per_image_metrics.csv"
    df.to_csv(csv_path,index=False)
    good=df[df.get("onnx_dice",pd.Series(dtype=float)).notna()].copy()
    if good.empty: raise RuntimeError("No images completed successfully. See console errors.")

    metric_cols=["pt_dice","pt_iou","pt_precision","pt_recall","onnx_dice","onnx_iou",
                 "onnx_precision","onnx_recall","pt_pixel_accuracy","onnx_pixel_accuracy"]
    summary={"dataset":"BRISC2025 segmentation test split",
             "images_found":len(images),"matched_pairs":len(pairs),
             "successful_images":len(good),"failed_images":int(len(df)-len(good)),
             "missing_masks":[Path(p).name for p in missing_masks],
             "checkpoint":str(ckpt_path),"onnx_model":str(ROOT/"models"/"deploy"/"segmenter.onnx"),
             "preprocessing_policy":policy,"image_size":size,"threshold":threshold,
             "empty_gt_images":int(good["onnx_gt_empty"].sum()),
             "empty_gt_false_positive_images":int(((good["onnx_gt_empty"]==1)&(good["onnx_pred_pixels"]>0)).sum()),
             "pt_onnx_mask_agreement_mean":float(np.mean(parity_agree)),
             "pt_onnx_mask_agreement_min":float(np.min(parity_agree)),
             "logit_max_abs_diff_mean":float(np.mean(parity_abs_max)),
             "logit_max_abs_diff_max":float(np.max(parity_abs_max)),
             "logit_mean_abs_diff_mean":float(np.mean(parity_abs_mean)),
             "relative_l2_diff_mean":float(np.mean(parity_rel)),
             "metrics":{}}
    for col in metric_cols:
        vals=good[col].dropna().to_numpy(dtype=float)
        summary["metrics"][col]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),
                                  "std":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,
                                  "bootstrap_95ci_mean":bootstrap_ci(vals)}
    by_class={}
    for cls, group in good.groupby("class",dropna=False):
        by_class[str(cls)]={"n":int(len(group)),
            "onnx_dice_mean":float(group["onnx_dice"].mean()),
            "onnx_dice_median":float(group["onnx_dice"].median()),
            "onnx_iou_mean":float(group["onnx_iou"].mean()),
            "onnx_precision_mean":float(group["onnx_precision"].mean()),
            "onnx_recall_mean":float(group["onnx_recall"].mean()),
            "empty_gt_n":int(group["onnx_gt_empty"].sum())}
    summary["by_filename_class"] = by_class
    json_path=OUT_DIR/"summary.json"
    json_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")

    # Plots are diagnostic, not a substitute for reporting per-image distributions.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(good["onnx_dice"].dropna(),bins=25)
    ax.set(title="ONNX per-image Dice distribution",xlabel="Dice",ylabel="Number of images")
    fig.tight_layout(); fig.savefig(OUT_DIR/"dice_distribution.png",dpi=160); plt.close(fig)
    class_order=list(by_class)
    fig,ax=plt.subplots(figsize=(9,5))
    if class_order:
        vals=[good.loc[good["class"]==c,"onnx_dice"].dropna().to_numpy() for c in class_order]
        ax.boxplot(vals,tick_labels=class_order,showmeans=True)
        ax.set(title="ONNX Dice by filename-derived group",ylabel="Per-image Dice")
    fig.tight_layout(); fig.savefig(OUT_DIR/"dice_by_class.png",dpi=160); plt.close(fig)
    worst=good.nsmallest(min(20,len(good)),"onnx_dice")
    worst[["image","mask","class","onnx_dice","onnx_iou","onnx_precision","onnx_recall",
           "onnx_gt_pixels","onnx_pred_pixels","onnx_fp","onnx_fn"]].to_csv(OUT_DIR/"worst_20_cases.csv",index=False)

    # Generate visual error analysis in the same run; no second script is required.
    create_error_analysis(good, core, process_mask, cfg, sess, in_name, policy, size, threshold)

    print("\n"+"="*78+"\nSUMMARY")
    print(f"Successful: {len(good)}/{len(pairs)} | Failed: {len(df)-len(good)} | Missing masks: {len(missing_masks)}")
    for col in ("onnx_dice","onnx_iou","onnx_precision","onnx_recall"):
        v=summary["metrics"][col]
        print(f"{col:18s} mean={v['mean']:.5f} median={v['median']:.5f} 95% CI=[{v['bootstrap_95ci_mean'][0]:.5f}, {v['bootstrap_95ci_mean'][1]:.5f}]")
    print(f"PyTorch/ONNX mean binary-mask agreement: {summary['pt_onnx_mask_agreement_mean']*100:.5f}%")
    print(f"Mean per-image max logit difference: {summary['logit_max_abs_diff_mean']:.7g}")
    print("\nGroups inferred from filename tokens (inspect counts before interpreting):")
    for c,v in by_class.items(): print(f"  {c}: n={v['n']}, mean Dice={v['onnx_dice_mean']:.5f}")
    print("\nOutputs:")
    for p in (csv_path,json_path,OUT_DIR/"dice_distribution.png",OUT_DIR/"dice_by_class.png",OUT_DIR/"worst_20_cases.csv",
              ERROR_DIR/"failure_case_metrics.csv",ERROR_DIR/"worst_cases_contact_sheet.png",
              ERROR_DIR/"full_testset_by_filename_class.csv",ERROR_DIR/"full_testset_by_orientation.csv"):
        print(f"  {p}")
    print("\nIMPORTANT: test-set metrics are final evaluation only. Do not tune threshold on this split.")
    print("Filename-derived class grouping is provisional; verify labels against the official dataset metadata.")
    print("All outputs were saved under the existing report/ folder; no folder consolidation was performed.")

if __name__ == "__main__":
    try: main()
    except Exception:
        print("\nFULL TEST-SET VALIDATION FAILED:\n"); traceback.print_exc()
        raise
