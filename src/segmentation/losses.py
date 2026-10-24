"""Segmentation losses. Inputs: logits (B,1,H,W) float32 (before sigmoid), target (B,1,H,W) float in {0,1}.

Tumors cover ~1.6% of pixels, so plain pixel accuracy (or unweighted BCE) is dominated by background.
  BCE   = -mean[ y log p + (1-y) log(1-p) ]                      p = sigmoid(logit)
  Dice  = 1 - (2*sum(p*y) + s) / (sum(p) + sum(y) + s)           soft (differentiable) Dice, per image, s = smoothing
          s = 1 makes an empty target with an empty prediction score a perfect 1, and penalises false alarms on empty targets.
  Focal = mean[ -alpha_t (1-p_t)^gamma log p_t ]                 p_t = p if y=1 else 1-p; down-weights easy pixels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)       # numerically stable sigmoid+BCE in one op


def dice_loss(logits, target, smooth=1.0):
    p = torch.sigmoid(logits).flatten(1)                            # (B, H*W)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    dice = (2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)    # one Dice per image
    return 1 - dice.mean()                                          # every image counts equally (matches the metric)


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")   # = -log p_t per pixel
    p_t = torch.exp(-bce)                                           # probability of the true class
    alpha_t = alpha * target + (1 - alpha) * (1 - target)           # class weight: alpha for tumor, 1-alpha for background
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


class SegLoss(nn.Module):
    """kind: 'bce_dice' (BRISC / Swin-HAFNet: BCE + Dice), 'dice_focal' (Diagnostics 2026: 0.7 Dice + 0.3 Focal), 'bce', 'dice'."""

    def __init__(self, kind="bce_dice"):
        super().__init__()
        assert kind in ("bce_dice", "dice_focal", "bce", "dice"), kind
        self.kind = kind

    def forward(self, logits, target):
        if self.kind == "bce":
            return bce_loss(logits, target)
        if self.kind == "dice":
            return dice_loss(logits, target)
        if self.kind == "bce_dice":
            return bce_loss(logits, target) + dice_loss(logits, target)
        return 0.7 * dice_loss(logits, target) + 0.3 * focal_loss(logits, target)
