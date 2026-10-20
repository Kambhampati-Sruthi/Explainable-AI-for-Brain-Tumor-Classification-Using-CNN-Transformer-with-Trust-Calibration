"""Augmentations for BRISC2025. Every geometric operation is applied to the image AND the mask
with ONE set of random numbers, so they can never drift apart.

Tensors: image = float32 (1, S, S) in [0, 1];  mask = float32 (1, S, S) with values 0/1 (or None).
Libraries: torch (math), torch.nn.functional (resampling), torchvision.transforms.functional (Gaussian blur).
Reference: Isensee et al. 2021, nnU-Net, Nature Methods (the augmentation families used here).
"""
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# All probabilities and ranges live in one dictionary, so ablations just change numbers here.
AUG_CFG = dict(
    p_flip=0.5,                                   # horizontal mirror
    p_affine=0.8, max_deg=10.0,                   # rotation in degrees (small: MRI has a standard orientation)
    max_shift=0.08,                               # translation as a fraction of image width/height
    scale_range=(0.70, 1.10),                     # <1 zooms OUT (head smaller, black borders), >1 zooms in
    p_croppad=0.30, min_frac=0.70,                # crop >= 70% of each side, pad back to square (shortcut breaker)
    p_gamma=0.5, gamma_range=(0.7, 1.4), gain_range=(0.8, 1.25),   # contrast curve and brightness
    p_blur=0.30, blur_sigma=(0.3, 1.5),           # Gaussian blur (pixels)
    p_downup=0.30, downup_factor=(0.5, 0.9),      # simulate a lower native resolution
    p_noise=0.30, noise_sigma=(0.005, 0.02),      # additive Gaussian noise, in [0,1] intensity units
)


def _rand(lo, hi):
    """Uniform random float in [lo, hi]. Uses torch's RNG, which DataLoader seeds per worker."""
    return lo + (hi - lo) * torch.rand(1).item()


def _chance(p):
    """True with probability p."""
    return torch.rand(1).item() < p


def _resize(img, mask, size):
    """Resize to size x size. Image: bilinear (+antialias when shrinking).
    Mask: bilinear on the 0/1 values, then threshold at 0.5 (the rule validated in Stage 2A)."""
    img = F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear",
                        align_corners=False, antialias=True).squeeze(0)
    if mask is not None:
        m = F.interpolate(mask.unsqueeze(0), size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True).squeeze(0)
        mask = (m > 0.5).float()
    return img, mask


def flip(img, mask):
    """Mirror left-right. dims=[-1] is the width axis."""
    img = torch.flip(img, dims=[-1])
    return img, (None if mask is None else torch.flip(mask, dims=[-1]))


def affine(img, mask, angle_deg, scale, tx, ty):
    """Rotate by angle, scale about the centre, translate by (tx, ty); empty areas become black (0).

    Math: grid_sample is a BACKWARD warp. For each OUTPUT pixel p_out it reads the INPUT at
        p_in = A (p_out - t),   A = (1/s) * R(-a),   R(-a) = [[cos a, sin a], [-sin a, cos a]]
    Then content at p_c lands at p_out = s R(a) p_c + t: rotated by a, scaled by s, shifted by t.
    Coordinates are normalised to [-1, 1], so the full image width is 2 units (tx = 2 * fraction).
    """
    _, H, W = img.shape
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    A = torch.tensor([[c / scale, s / scale], [-s / scale, c / scale]], dtype=torch.float32)
    t = torch.tensor([tx, ty], dtype=torch.float32)
    b = -A @ t                                                    # offset so that p_in = A p_out + b = A (p_out - t)
    theta = torch.cat([A, b.unsqueeze(1)], dim=1).unsqueeze(0)    # (1, 2, 3): the matrix affine_grid expects
    grid = F.affine_grid(theta, size=(1, 1, H, W), align_corners=False)   # sampling coordinates for every output pixel
    out_img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False).squeeze(0)
    out_mask = None
    if mask is not None:                                          # same grid -> same warp -> aligned
        m = F.grid_sample(mask.unsqueeze(0), grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False).squeeze(0)
        out_mask = (m > 0.5).float()                              # re-binarize
    return out_img, out_mask


def random_crop_pad(img, mask, min_frac):
    """Crop a random rectangle (each side >= min_frac of the image), centre it on a black square
    canvas of side max(cw, ch), then resize back. Creates portrait/landscape-with-bars images
    in ALL classes, so black bars stop being a class cue."""
    _, H, W = img.shape
    cw = max(16, int(round(_rand(min_frac, 1.0) * W)))           # crop width in pixels
    ch = max(16, int(round(_rand(min_frac, 1.0) * H)))           # crop height in pixels
    left = int(_rand(0, W - cw))                                 # random crop position (int() = floor)
    top = int(_rand(0, H - ch))
    side = max(cw, ch)                                           # canvas side keeps aspect ratio (no distortion)
    t0, l0 = (side - ch) // 2, (side - cw) // 2                  # offsets that centre the crop on the canvas

    def crop_and_pad(x):
        canvas = torch.zeros(1, side, side, dtype=x.dtype)       # black canvas
        canvas[:, t0:t0 + ch, l0:l0 + cw] = x[:, top:top + ch, left:left + cw]
        return canvas

    img_p = crop_and_pad(img)
    mask_p = None if mask is None else crop_and_pad(mask)
    return _resize(img_p, mask_p, H)


def geometric(img, mask, cfg=AUG_CFG):
    """All spatial augmentations (image and mask together)."""
    if _chance(cfg["p_flip"]):
        img, mask = flip(img, mask)
    if _chance(cfg["p_affine"]):
        angle = _rand(-cfg["max_deg"], cfg["max_deg"])
        scale = _rand(*cfg["scale_range"])
        tx = _rand(-cfg["max_shift"], cfg["max_shift"]) * 2.0    # *2: normalised width is 2 units
        ty = _rand(-cfg["max_shift"], cfg["max_shift"]) * 2.0
        img, mask = affine(img, mask, angle, scale, tx, ty)
    if _chance(cfg["p_croppad"]):
        img, mask = random_crop_pad(img, mask, cfg["min_frac"])
    return img, mask


def photometric(img, cfg=AUG_CFG):
    """Intensity/resolution augmentations: change pixel values only, never geometry, so the mask is untouched."""
    S = img.shape[-1]
    if _chance(cfg["p_gamma"]):
        gamma = _rand(*cfg["gamma_range"])                       # x -> x^gamma bends the contrast curve
        gain = _rand(*cfg["gain_range"])                         # global brightness factor
        img = (img.clamp(0, 1) ** gamma) * gain
    if _chance(cfg["p_blur"]):
        sigma = _rand(*cfg["blur_sigma"])
        k = 2 * math.ceil(3 * sigma) + 1                         # kernel covers +-3 sigma and must be odd
        img = TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])
    if _chance(cfg["p_downup"]):                                 # shrink then enlarge = lost fine detail
        f = _rand(*cfg["downup_factor"])
        small = max(16, int(S * f))
        img = F.interpolate(img.unsqueeze(0), size=(small, small), mode="bilinear",
                            align_corners=False, antialias=True)
        img = F.interpolate(img, size=(S, S), mode="bilinear", align_corners=False).squeeze(0)
    if _chance(cfg["p_noise"]):
        img = img + torch.randn_like(img) * _rand(*cfg["noise_sigma"])
    return img.clamp(0.0, 1.0)                                   # keep valid intensity range


class TrainAugment:
    """Callable used by the Dataset: (img, mask) -> (img, mask). Toggle groups for ablations."""

    def __init__(self, cfg=None, geometric_on=True, photometric_on=True):
        self.cfg = dict(AUG_CFG if cfg is None else cfg)
        self.geometric_on, self.photometric_on = geometric_on, photometric_on

    def __call__(self, img, mask=None):
        if self.geometric_on:
            img, mask = geometric(img, mask, self.cfg)
        if self.photometric_on:
            img = photometric(img, self.cfg)
        return img, mask
