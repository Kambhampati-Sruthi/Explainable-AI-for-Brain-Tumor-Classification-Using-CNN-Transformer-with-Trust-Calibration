"""Segmentation metrics, computed PER IMAGE then averaged (so every image counts equally).

  Dice      = 2TP / (2TP + FP + FN)          IoU = TP / (TP + FP + FN)          (IoU = Dice / (2 - Dice) per image)
  Precision = TP / (TP + FP)                 Recall = TP / (TP + FN)
  HD95      : boundary distance. d(P -> G) = distance from each boundary pixel of P to the nearest boundary pixel of G.
              HD95 = max( 95th percentile of d(P->G), 95th percentile of d(G->P) )   (the convention used by MONAI).
              The 95th percentile (not the maximum) keeps one stray pixel from dominating. Units = pixels at 256 px
              (multiply by ~2 for a 512-px equivalent). Undefined when either mask is empty (-> NaN; counted as "missed").
Images whose ground truth is EMPTY (no_tumor images, and the one 24-pixel mask that vanishes at 256 px) are excluded
from tumor metrics and reported separately.
"""
import numpy as np
import pandas as pd
import torch
from scipy import ndimage


def per_image_metrics(pred, gt):
    """pred, gt: bool tensors (B,1,H,W). Returns dict of float tensors (B,). 0/0 gives NaN, handled by the callers."""
    p, g = pred.flatten(1), gt.flatten(1)
    tp = (p & g).sum(1).float()
    fp = (p & ~g).sum(1).float()
    fn = (~p & g).sum(1).float()
    return {"dice": 2 * tp / (2 * tp + fp + fn), "iou": tp / (tp + fp + fn),
            "precision": tp / (tp + fp), "recall": tp / (tp + fn)}


def _surface(mask):
    """Boundary pixels of a boolean 2-D mask: the mask minus its erosion (border_value=0 keeps objects touching the edge)."""
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def hd95(pred, gt):
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_g = ndimage.distance_transform_edt(~sg)          # distance of every pixel to the nearest GT boundary pixel
    d_to_p = ndimage.distance_transform_edt(~sp)
    return float(max(np.percentile(d_to_g[sp], 95), np.percentile(d_to_p[sg], 95)))


@torch.no_grad()
def predict_probs(model, loader, device):
    """Returns (probs (N,1,H,W) float32 CPU, masks (N,1,H,W) float32 CPU, cache_row_index (N,))."""
    model.eval()
    probs, masks, idx = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(b["image"].to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu())
        masks.append(b["mask"])
        idx.append(b["index"])
    return torch.cat(probs), torch.cat(masks), torch.cat(idx).numpy()


def evaluate_seg(model, loader, device, thr=0.5, with_hd=True):
    """Per-image table (one row per image) plus the probability maps."""
    probs, masks, idx = predict_probs(model, loader, device)
    pred, gt = probs > thr, masks > 0.5
    df = pd.DataFrame({k: v.numpy() for k, v in per_image_metrics(pred, gt).items()})
    df["index"] = idx
    df["gt_px"] = gt.flatten(1).sum(1).numpy()
    df["pred_px"] = pred.flatten(1).sum(1).numpy()
    if with_hd:
        pn, gn = pred[:, 0].numpy(), gt[:, 0].numpy()
        df["hd95"] = [hd95(pn[i], gn[i]) for i in range(len(df))]
    return df, probs


def summarize(df):
    """Tumor-image summary. precision/recall use pandas' NaN-skipping mean; 'missed' = tumors with an EMPTY prediction."""
    d = df[df["gt_px"] > 0]
    tp = d["dice"] * (d["gt_px"] + d["pred_px"]) / 2                      # recover TP from Dice
    out = {"n": len(d), "dice": d["dice"].mean(), "iou": d["iou"].mean(),
           "global_iou": tp.sum() / (d["gt_px"].sum() + d["pred_px"].sum() - tp.sum()),   # pixel-aggregated IoU
           "precision": d["precision"].mean(), "recall": d["recall"].mean(), "missed": int((d["pred_px"] == 0).sum())}
    if "hd95" in d:
        out.update(hd95_mean=d["hd95"].mean(), hd95_median=d["hd95"].median())
    return out


@torch.no_grad()
def quick_eval(model, loader, device, loss_fn):
    """Cheap validation used every epoch: mean loss, mean per-image Dice and IoU (tumor images only)."""
    model.eval()
    tot, n, dice, iou = 0.0, 0, [], []
    for b in loader:
        x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        tot += loss_fn(logits.float(), y).item() * len(x)
        n += len(x)
        m = per_image_metrics(torch.sigmoid(logits.float()) > 0.5, y > 0.5)
        keep = y.flatten(1).sum(1) > 0                                   # skip empty-GT images
        dice.append(m["dice"][keep]); iou.append(m["iou"][keep])
    return tot / n, torch.cat(dice).mean().item(), torch.cat(iou).mean().item()
