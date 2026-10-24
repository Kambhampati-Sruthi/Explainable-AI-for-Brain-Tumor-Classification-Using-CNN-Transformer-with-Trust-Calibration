"""Torch-free rendering/measurement helpers for the web app (all layers are 512x512 and perfectly aligned)."""
import base64, io
import numpy as np, cv2
from PIL import Image

OUT = 512


def b64_png(arr):
    buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def base_layer(arr_u8):
    return cv2.resize(arr_u8, (OUT, OUT), interpolation=cv2.INTER_CUBIC)


def heat_rgba(cam, gamma=0.8):
    """Jet-coloured heat-map whose opacity grows with attention (low attention stays transparent)."""
    import matplotlib
    c = np.clip(cv2.resize(cam.astype(np.float32), (OUT, OUT), interpolation=cv2.INTER_CUBIC), 0, 1)
    rgba = (matplotlib.colormaps["jet"](c) * 255).astype(np.uint8); rgba[..., 3] = (np.power(c, gamma) * 255).astype(np.uint8)
    return rgba


def mask_layers(mask):
    """-> (translucent fill RGBA, outline RGBA) of a boolean mask, smooth at 512 px."""
    m = cv2.resize(mask.astype(np.float32), (OUT, OUT), interpolation=cv2.INTER_LINEAR) > 0.5
    fill = np.zeros((OUT, OUT, 4), np.uint8); fill[m] = (34, 197, 94, 95)
    edge_px = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    edge = np.zeros((OUT, OUT, 4), np.uint8); edge[edge_px] = (60, 255, 130, 255)
    return fill, edge


def overlay_rgb(arr_u8, cam=None, mask=None):
    """Flattened overlay (gray + heat-map + outline) used in the 'compare' grid, 512x512 RGB."""
    rgb = np.repeat(base_layer(arr_u8)[..., None], 3, 2).astype(np.float32)
    if cam is not None:
        h = heat_rgba(cam).astype(np.float32); a = (h[..., 3:4] / 255.0) * 0.65; rgb = rgb * (1 - a) + h[..., :3] * a
    if mask is not None and mask.any():
        _, edge = mask_layers(mask); e = edge[..., 3] > 0; rgb[e] = edge[e][:, :3]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def measure(mask, size=256):
    """Region statistics in the 256x256 analysis frame (no physical units: DICOM pixel spacing is unknown)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return dict(present=False)
    return dict(present=True, bbox=dict(x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()), w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1)),
                centroid=dict(x=round(float(xs.mean()), 1), y=round(float(ys.mean()), 1)), frame=size)