import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import ndimage
from scipy.stats import spearmanr


def del_ins(model, x, heat, target, base, patch=16, n_steps=17):
    """Deletion / insertion curves (Petsiuk et al. 2018). Patches are ranked by attribution; deletion replaces the top-k patches
    by the black baseline (class probability should DROP fast -> low AUC); insertion starts from a blurred image and restores
    top-k patches (probability should RISE fast -> high AUC). AUC over the fraction of patches changed."""
    S_ = x.shape[-1]; P = S_ // patch
    score = F.avg_pool2d(torch.as_tensor(heat, device=x.device).float()[None, None], patch).flatten()
    order = torch.argsort(score, descending=True)
    ks = torch.linspace(0, P * P, n_steps, device=x.device).long()
    m = torch.zeros(n_steps, P * P, device=x.device)
    for i, k in enumerate(ks): m[i, order[:k]] = 1
    M = F.interpolate(m.view(n_steps, 1, P, P), scale_factor=patch, mode="nearest")
    blurred = TF.gaussian_blur(x, [31, 31], [10.0, 10.0])
    with torch.no_grad():
        p_del = torch.softmax(model(x * (1 - M) + base * M), 1)[:, target]
        p_ins = torch.softmax(model(blurred * (1 - M) + x * M), 1)[:, target]
    xs = (ks.float() / (P * P)).cpu().numpy()
    return float(np.trapz(p_del.cpu().numpy(), xs)), float(np.trapz(p_ins.cpu().numpy(), xs))


def pool(h, k=8):
    return F.avg_pool2d(torch.as_tensor(h).float()[None, None], k)[0, 0].flatten().cpu().numpy()      # 32x32 grid for stable rank stats

def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else np.nan

def rel_change(a, b):
    a, b = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
    return float(np.linalg.norm(a - b))                                                           # distance between unit-norm maps

def energy(h, mask):
    h = np.clip(np.asarray(h, float), 0, None); t = h.sum()
    return float((h * mask).sum() / t) if t > 0 else np.nan

def pointing_hit(h, mask, tol=8, sigma=2.0):
    hs = ndimage.gaussian_filter(np.clip(np.asarray(h, float), 0, None), sigma)
    y, x = np.unravel_index(np.argmax(hs), hs.shape)
    return bool((ndimage.distance_transform_edt(~mask) <= tol)[y, x])                              # inside the mask dilated by tol px

def iou_at(h, mask, frac=0.15):
    h = np.asarray(h, float); b = h >= frac * h.max()
    return float((b & mask).sum() / max((b | mask).sum(), 1))

def center_map(size=256, sigma=64.0):
    g = np.arange(size) - size / 2; return np.exp(-(g[None, :] ** 2 + g[:, None] ** 2) / (2 * sigma ** 2))

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); prev = 0.0
    for r, i in enumerate(order):
        prev = max(prev, (m - r) * pvals[i]); adj[i] = min(prev, 1.0)
    return adj
