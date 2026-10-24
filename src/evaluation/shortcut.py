"""Format-shift (shortcut sensitivity) test.
Take TUMOR images and change only their format look (borders / zoom). A model that decides by tumor appearance keeps
its answer; a model that uses "odd format => no_tumor" starts answering no_tumor. Interventions run at TEST time on
the validation images, through the Dataset's `augment` hook, i.e. BEFORE normalization (black stays black).
Images are float (1, S, S) in [0, 1]; the mask argument is unused here (None for classification) but kept for the API.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.preprocessing import augment as A
from src.preprocessing.dataset import BrainMRIDataset, CLASS_ORDER
from src.classification.evaluate import predict

NO_TUMOR = CLASS_ORDER.index("no_tumor")
KEEP = 0.585          # kept central width fraction = 216/369, the proportions of the commonest portrait images


def _strip(S):
    w = int(round(KEEP * S))
    return w, (S - w) // 2            # kept width and its left offset (strip is centred)


def identity(img, mask):
    return img, mask


def mirror(img, mask):                # control: mirroring should not change the diagnosis
    return A.flip(img, mask)


def zoom_out(img, mask):              # whole image shrinks to 70%, black borders on all four sides
    return A.affine(img, mask, 0.0, 0.7, 0.0, 0.0)


def side_crop(img, mask):             # keep only the central strip; sides become black (outer anatomy removed)
    S = img.shape[-1]
    w, l = _strip(S)
    out = torch.zeros_like(img)
    out[:, :, l:l + w] = img[:, :, l:l + w]
    return out, mask


def squeeze(img, mask):               # compress the whole image horizontally into the strip (shape distorted)
    S = img.shape[-1]
    w, l = _strip(S)
    narrow = F.interpolate(img.unsqueeze(0), size=(S, w), mode="bilinear", align_corners=False, antialias=True).squeeze(0)
    out = torch.zeros_like(img)
    out[:, :, l:l + w] = narrow
    return out, mask


INTERVENTIONS = {"identity": identity, "mirror": mirror, "zoom_out": zoom_out, "side_crop": side_crop, "squeeze": squeeze}


def eligible_positions(ds, M):
    """Dataset positions of TUMOR images whose whole mask lies inside the kept central strip. Using the same
    subset for every intervention keeps the comparison fair and guarantees the tumor evidence is never removed."""
    S = M.shape[-1]
    w, l = _strip(S)
    pos = []
    for i, rm in enumerate(ds.row_mask):
        if rm < 0:
            continue                                      # no_tumor image: no mask, not part of this test
        m = M[rm]
        if m[:, :l].sum() + m[:, l + w:].sum() == 0:      # no tumor pixel outside the strip
            pos.append(i)
    return pos


def format_shift_test(model, X, M, cache_index, norm, stats, device, part="val", batch_size=64):
    """Returns {intervention: {n, acc, nt_k, nt_rate}} on the eligible tumor images of `part`."""
    base = BrainMRIDataset(X, M, cache_index, part, task="cls", norm=norm, dataset_stats=stats)
    pos = eligible_positions(base, M)
    out = {}
    for name, fn in INTERVENTIONS.items():
        ds = BrainMRIDataset(X, M, cache_index, part, task="cls", norm=norm, dataset_stats=stats, augment=fn)
        loader = DataLoader(Subset(ds, pos), batch_size=batch_size, shuffle=False, num_workers=0)
        logits, labels, _ = predict(model, loader, device)
        pred = logits.argmax(1)
        out[name] = {"n": len(pred), "acc": float((pred == labels).mean()),
                     "nt_k": int((pred == NO_TUMOR).sum()), "nt_rate": float((pred == NO_TUMOR).mean())}
    return out


def format_shift_test_v2(model, X, M, cache_index, norm, stats, device, part="test", test_set="dedup", batch_size=64):
    """Same as format_shift_test, but the TEST partition can be restricted to the de-duplicated subset."""
    kw = dict(task="cls", norm=norm, dataset_stats=stats, test_set=test_set)
    pos = eligible_positions(BrainMRIDataset(X, M, cache_index, part, **kw), M)
    out = {}
    for name, fn in INTERVENTIONS.items():
        ds = BrainMRIDataset(X, M, cache_index, part, augment=fn, **kw)
        logits, labels, _ = predict(model, DataLoader(Subset(ds, pos), batch_size=batch_size, shuffle=False, num_workers=0), device)
        pred = logits.argmax(1)
        out[name] = {"n": len(pred), "acc": float((pred == labels).mean()),
                     "nt_k": int((pred == NO_TUMOR).sum()), "nt_rate": float((pred == NO_TUMOR).mean())}
    return out
