"""XAI methods for the EfficientNet-B0 classifier (hand-written Grad-CAM + Captum wrappers + SiLU-guided backprop)."""
import copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import ndimage


# ---------- inputs ----------
def norm_stats(norm, stats, device):
    from src.preprocessing.dataset import IMAGENET_MEAN, IMAGENET_STD
    if norm == "imagenet": mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif norm == "dataset": mean, std = (stats[0],) * 3, (stats[1],) * 3
    else: mean, std = (0.0,) * 3, (1.0,) * 3
    return torch.tensor(mean, device=device).view(1, 3, 1, 1), torch.tensor(std, device=device).view(1, 3, 1, 1)

def to_input_batch(imgs_u8, norm, stats, device):
    """uint8 (B,S,S) -> normalized float (B,3,S,S), exactly like the Dataset (gray replicated to 3 channels)."""
    x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).float().div(255).to(device)[:, None].expand(-1, 3, -1, -1)
    mean, std = norm_stats(norm, stats, device)
    return ((x - mean) / std).contiguous()

def black_baseline(norm, stats, size, device):
    """An all-black image expressed in NORMALIZED space (the 'absence of signal' reference for IG / Occlusion / deletion)."""
    mean, std = norm_stats(norm, stats, device)
    return ((0.0 - mean) / std).expand(1, 3, size, size).contiguous()


# ---------- Grad-CAM (Selvaraju et al. 2017), written by hand, any number of layers in ONE forward/backward ----------
class MultiGradCAM:
    def __init__(self, model, layers):                      # layers: {"L8": model.features[8], ...}
        self.model, self.names, self.acts, self.enabled = model, list(layers), {}, True
        self.handles = [m.register_forward_hook(self._mk(n)) for n, m in layers.items()]
    def _mk(self, n):
        def hook(mod, inp, out):
            if self.enabled:
                self.acts[n] = out                           # keep the layer output A^k (it is part of the autograd graph)
        return hook
    def remove(self):
        for h in self.handles: h.remove()
    def __call__(self, x, target=None, size=256, normalize=True):
        logits = self.model(x)
        if target is None: target = logits.argmax(1)
        target = torch.as_tensor(target, device=x.device).view(-1)
        if target.numel() == 1 and len(x) > 1: target = target.expand(len(x))
        score = logits.gather(1, target[:, None]).sum()      # y^c BEFORE softmax; samples are independent in eval mode
        acts = [self.acts[n] for n in self.names]
        grads = torch.autograd.grad(score, acts)             # dy^c/dA^k_ij  (backward hooks are avoided: SiLU is in-place)
        out = {}
        for n, a, g in zip(self.names, acts, grads):
            alpha = g.mean(dim=(2, 3), keepdim=True)         # (1/Z) sum_ij  -> neuron importance weights alpha_k^c
            cam = F.relu((alpha * a).sum(1, keepdim=True))   # ReLU(sum_k alpha_k A^k): keep only positive influence
            cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
            if normalize:
                cam = cam / cam.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            out[n] = cam[:, 0].detach()
        return out, logits.detach(), target


# ---------- Guided Backprop adapted to SiLU ----------
class _GuidedSiLUFn(torch.autograd.Function):
    """Forward = SiLU(x) = x*sigmoid(x). Backward = clamp(g,0) * clamp(SiLU'(x),0): only positive gradients flow through
    positive-derivative regions. For ReLU this is exactly guided backprop; for SiLU it is OUR generalisation."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return g.clamp(min=0) * (s * (1 + x * (1 - s))).clamp(min=0)

class GuidedSiLU(nn.Module):
    def forward(self, x): return _GuidedSiLUFn.apply(x)

def make_guided_model(model):
    m, n = copy.deepcopy(model), [0]
    def swap(mod):
        for name, child in mod.named_children():
            if isinstance(child, nn.SiLU): setattr(mod, name, GuidedSiLU()); n[0] += 1
            else: swap(child)
    swap(m)
    return m.eval(), n[0]

def guided_backprop(guided_model, x, target):
    x = x.clone().requires_grad_(True)
    score = guided_model(x).gather(1, torch.as_tensor(target, device=x.device).view(-1, 1)).sum()
    return torch.autograd.grad(score, x)[0]                  # (B,3,S,S)

def plain_gradient(model, x, target):
    x = x.clone().requires_grad_(True)
    score = model(x).gather(1, torch.as_tensor(target, device=x.device).view(-1, 1)).sum()
    return torch.autograd.grad(score, x)[0]


# ---------- Captum wrappers ----------
def ig_attr(model, x, target, baseline, n_steps=50, internal_bs=25):
    from captum.attr import IntegratedGradients
    return IntegratedGradients(model).attribute(x, baselines=baseline, target=int(target), n_steps=n_steps,
                                                internal_batch_size=internal_bs, return_convergence_delta=True)

def occlusion_attr(model, x, target, baseline, win=32, stride=16):
    from captum.attr import Occlusion
    return Occlusion(model).attribute(x, sliding_window_shapes=(3, win, win), strides=(3, stride, stride),
                                      baselines=baseline, target=int(target), perturbations_per_eval=16)


# ---------- helpers shared by Stages 5-8 ----------
def overlay(img_u8, heat, alpha=0.45):
    import matplotlib
    h = np.asarray(heat, float); h = (h - h.min()) / max(h.max() - h.min(), 1e-12)
    rgb = matplotlib.colormaps["jet"](h)[..., :3]
    base = np.repeat(img_u8[..., None] / 255.0, 3, axis=2)
    return (1 - alpha) * base + alpha * rgb

def energy_batch(cam, mask):
    """Fraction of (non-negative) attribution mass inside a boolean mask, per image. cam, mask: (B,S,S)."""
    tot = cam.sum((1, 2))
    return (cam * mask).sum((1, 2)) / np.where(tot > 0, tot, np.nan)
