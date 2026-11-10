"""Segmentation-specific XAI for the project's EfficientNet-B0 U-Net++.

Methods:
  * target-specific Grad-CAM on decoder node 0_1
  * Integrated Gradients for a fixed output ROI
  * Occlusion sensitivity for a fixed output ROI

Targets are explicit: predicted, ground_truth (offline audit only), or custom mask.
Ground-truth targeting is an analysis oracle and MUST NOT be used in live inference.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _norm01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0)
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def _roi_tensor(roi, device, shape):
    r = torch.as_tensor(roi, device=device, dtype=torch.float32)
    if r.ndim == 2: r = r[None, None]
    elif r.ndim == 3: r = r[:, None]
    if r.shape[-2:] != shape[-2:]:
        r = F.interpolate(r, size=shape[-2:], mode="nearest")
    return (r > 0.5).float()


class SegmentationXAI:
    """Explain a scalar score derived from a selected region of the segmenter output."""
    def __init__(self, model: torch.nn.Module, target_layer=None):
        self.model = model.eval()
        # U-Net++ node 0_1 is a high-resolution decoder representation.
        self.target_layer = target_layer or self.model.nodes["0_1"]
        self._activation = None
        self._handle = self.target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._activation = output

    def close(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None

    @staticmethod
    def score(logits, roi):
        """Mean logit in ROI; stable scalar target for attribution."""
        r = _roi_tensor(roi, logits.device, logits.shape)
        denom = r.sum().clamp_min(1.0)
        return (logits * r).sum() / denom

    def gradcam(self, x, roi):
        """Decoder Grad-CAM for evidence supporting the selected output region."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x.detach().clone().requires_grad_(True))
            score = self.score(logits, roi)
            if self._activation is None:
                raise RuntimeError("Target layer hook did not capture an activation")
            grad = torch.autograd.grad(score, self._activation, retain_graph=False)[0]
            weights = grad.mean(dim=(-2, -1), keepdim=True)
            cam = F.relu((weights * self._activation).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return _norm01(cam).detach()[0, 0].cpu().numpy(), float(score.detach().cpu())

    def integrated_gradients(self, x, roi, baseline=None, steps=32):
        """Riemann approximation to IG. Returns signed channel-summed attribution."""
        if steps < 2: raise ValueError("steps must be >= 2")
        x = x.detach()
        base = torch.zeros_like(x) if baseline is None else baseline.to(x.device, x.dtype)
        delta = x - base
        grads = []
        with torch.enable_grad():
            for alpha in torch.linspace(0, 1, steps, device=x.device):
                xi = (base + alpha * delta).detach().requires_grad_(True)
                logits = self.model(xi)
                s = self.score(logits, roi)
                grads.append(torch.autograd.grad(s, xi)[0].detach())
        # trapezoidal integration
        g = torch.stack(grads)
        avg = (g[0] * 0.5 + g[-1] * 0.5 + g[1:-1].sum(0)) / (steps - 1)
        attr = (delta * avg).sum(1, keepdim=True)
        return attr[0, 0].detach().cpu().numpy(), float(attr.sum().detach().cpu())

    @torch.no_grad()
    def occlusion(self, x, roi, window=32, stride=16, baseline_value=None):
        """Patch occlusion score-drop map. Positive values indicate evidence removed by occlusion."""
        if window <= 0 or stride <= 0: raise ValueError("window and stride must be positive")
        h, w = x.shape[-2:]
        base_score = self.score(self.model(x), roi).item()
        if baseline_value is None:
            baseline_value = (0.0, 0.0, 0.0)
        accum = torch.zeros((h, w), device=x.device)
        counts = torch.zeros_like(accum)
        for top in range(0, max(1, h - window + 1), stride):
            for left in range(0, max(1, w - window + 1), stride):
                y2, x2 = min(h, top + window), min(w, left + window)
                pert = x.clone()
                if isinstance(baseline_value, (tuple, list)):
                    bv = torch.as_tensor(baseline_value, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
                    pert[:, :, top:y2, left:x2] = bv
                else:
                    pert[:, :, top:y2, left:x2] = baseline_value
                drop = base_score - self.score(self.model(pert), roi).item()
                accum[top:y2, left:x2] += drop
                counts[top:y2, left:x2] += 1
        return (accum / counts.clamp_min(1)).detach().cpu().numpy()

    def explain(self, x, roi, baseline=None, occlusion_baseline=None, ig_steps=32, occlusion_window=32, occlusion_stride=16):
        cam, score = self.gradcam(x, roi)
        ig, ig_sum = self.integrated_gradients(x, roi, baseline=baseline, steps=ig_steps)
        occ = self.occlusion(x, roi, window=occlusion_window, stride=occlusion_stride, baseline_value=occlusion_baseline)
        return {"gradcam": cam, "integrated_gradients": ig, "occlusion": occ,
                "target_score": score, "ig_attribution_sum": ig_sum}
