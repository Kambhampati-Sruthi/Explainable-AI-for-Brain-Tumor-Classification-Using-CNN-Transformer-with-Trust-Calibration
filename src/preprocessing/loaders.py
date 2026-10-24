"""DataLoader factory. One function builds train/val/test loaders with consistent, reproducible settings."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import BrainMRIDataset


def seed_worker(worker_id):
    """Runs once inside every worker process.
    Inside a worker, torch.initial_seed() = base_seed + worker_id (derived from the loader's generator).
    PyTorch already seeds torch's RNG (used by our augmentations); we also seed numpy/random so any
    library code that uses them is reproducible too."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def class_weights(labels, n_classes=4):
    """Inverse-frequency weights.  w_c = N / (K * n_c):  a perfectly balanced set gives w = 1 for every class;
    rarer classes get w > 1. Returns (counts, weights)."""
    counts = np.bincount(labels, minlength=n_classes)
    return counts, len(labels) / (n_classes * counts)


def make_loaders(X, M, index_df, task, batch_size, num_workers, stats, norm="dataset", seed=42,
                 aug=None, include_negatives=False, use_sampler=False, eval_batch_mult=2):
    """Returns {"train","val","test","test_dedup","datasets"}.

    batch_size       : training batch size (evaluation uses batch_size * eval_batch_mult: no gradients = less memory)
    num_workers      : parallel CPU processes for loading/augmentation (0 = main process)
    aug              : augmentation callable for TRAIN only (val/test never get augmentation)
    use_sampler      : True = WeightedRandomSampler (class-balanced batches); False = ordinary shuffling
    """
    pin = torch.cuda.is_available()                      # pinned (page-locked) host memory speeds CPU->GPU copies
    base = dict(num_workers=num_workers, pin_memory=pin,
                worker_init_fn=seed_worker if num_workers > 0 else None)

    def ds(part, **kw):
        return BrainMRIDataset(X, M, index_df, part, task=task, norm=norm, dataset_stats=stats,
                               include_negatives=include_negatives, **kw)

    datasets = {"train": ds("train", augment=aug), "val": ds("val"),
                "test": ds("test", test_set="official"), "test_dedup": ds("test", test_set="dedup")}

    g = torch.Generator()                                # own RNG for shuffling => reproducible epoch order
    g.manual_seed(seed)
    train_kw = dict(base)
    if num_workers > 0:
        train_kw.update(persistent_workers=True, prefetch_factor=2)   # keep workers alive between epochs
    if use_sampler:
        counts, w = class_weights(datasets["train"].labels)
        sample_w = torch.as_tensor(w[datasets["train"].labels], dtype=torch.double)   # one weight per sample
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True, generator=g)
        train = DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler,
                           drop_last=True, generator=g, **train_kw)
    else:
        train = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                           drop_last=True, generator=g, **train_kw)   # drop_last: no tiny final batch (BatchNorm noise)

    ev_bs = batch_size * eval_batch_mult
    out = {"train": train, "datasets": datasets}
    for name in ["val", "test", "test_dedup"]:           # evaluation: fixed order, nothing dropped
        out[name] = DataLoader(datasets[name], batch_size=ev_bs, shuffle=False, drop_last=False, **base)
    return out
