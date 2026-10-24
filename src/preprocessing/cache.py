"""Offline preprocessing for BRISC2025: make images square, resize, cache as uint8 arrays."""
import cv2                       # OpenCV: fast, well-controlled resizing with explicit interpolation
import numpy as np
from PIL import Image            # used for decoding so results match the earlier duplicate audit
from tqdm.auto import tqdm       # progress bar


def read_gray(path):
    """Decode any image file to a 2-D uint8 array (H, W).

    convert("L") gives one channel. The 5,061 RGB-stored images were verified to have
    R == G == B in every pixel, so this loses no information.
    """
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def make_square(arr, policy):
    """Turn an (H, W) array into a square one. Works for images AND (float) masks.

    pad         : black borders on the short side, image centred. Nothing lost, no distortion.
    center_crop : keep the central min(H, W) square. No distortion, outer content lost.
    stretch     : leave unchanged here; resize() later squeezes it to size x size (distorts shape).
    """
    h, w = arr.shape
    if policy == "pad":
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=arr.dtype)      # black square (value 0)
        top, left = (side - h) // 2, (side - w) // 2          # offsets that centre the image
        canvas[top:top + h, left:left + w] = arr
        return canvas
    if policy == "center_crop":
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        return arr[top:top + side, left:left + side]
    if policy == "stretch":
        return arr
    raise ValueError(f"unknown policy: {policy}")


def _interp(h, w, size):
    """Shrinking -> INTER_AREA (averages source pixels = anti-aliasing, avoids jagged edges).
    Enlarging -> INTER_LINEAR (smooth interpolation). Chosen from the Stage 2A experiment."""
    return cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR


def process_image(arr, policy, size):
    """uint8 (H, W) MRI -> uint8 (size, size)."""
    sq = make_square(arr, policy)
    h, w = sq.shape
    return cv2.resize(sq, (size, size), interpolation=_interp(h, w, size))


def process_mask(arr, policy, size):
    """Mask (values 0-10 / 246-255) -> uint8 (size, size) holding only 0 and 1.

    Order matters: binarize at 127 (proven safe in Stage 1) -> make square -> resize the
    0/1 float mask with area averaging -> threshold at 0.5. Plain nearest-neighbour was
    shown to be unreliable at non-integer scale factors (Stage 2A).
    """
    fg = (arr > 127).astype(np.float32)
    sq = make_square(fg, policy)
    h, w = sq.shape
    out = cv2.resize(sq, (size, size), interpolation=_interp(h, w, size))
    return (out > 0.5).astype(np.uint8)


def build_cache(index_df, policy="pad", size=256):
    """Process every row of index_df (needs columns image_path and mask_path; mask_path may be NaN).

    Returns
      X        : (N, size, size) uint8 images, one per row
      M        : (K, size, size) uint8 masks, only for rows that have a mask (K <= N)
      mask_row : (N,) int64; mask_row[i] = row of M for image i, or -1 when image i has no mask
    """
    n = len(index_df)
    has_mask = index_df["mask_path"].notna().to_numpy()
    X = np.zeros((n, size, size), dtype=np.uint8)
    M = np.zeros((int(has_mask.sum()), size, size), dtype=np.uint8)
    mask_row = np.full(n, -1, dtype=np.int64)
    k = 0
    for i, row in enumerate(tqdm(index_df.itertuples(), total=n, desc=f"cache {policy} {size}")):
        X[i] = process_image(read_gray(row.image_path), policy, size)
        if has_mask[i]:
            M[k] = process_mask(read_gray(row.mask_path), policy, size)
            mask_row[i] = k
            k += 1
    return X, M, mask_row
