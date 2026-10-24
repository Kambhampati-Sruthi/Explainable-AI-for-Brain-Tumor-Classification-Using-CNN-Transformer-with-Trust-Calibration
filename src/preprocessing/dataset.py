"""PyTorch Dataset over the uint8 cache. One class serves classification, segmentation, or both."""
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_ORDER = ["glioma", "meningioma", "no_tumor", "pituitary"]        # alphabetical, fixed everywhere
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}               # string label -> integer 0..3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BrainMRIDataset(Dataset):
    """
    X, M          : cache arrays (N,S,S) and (K,S,S) uint8, loaded once into RAM
    index_df      : the cache_index table (columns row_img, row_mask, partition, tumor_label, ...)
    partition     : "train" | "val" | "test"
    task          : "cls" (image,label) | "seg" (image,mask) | "both"
    augment       : callable (img, mask) -> (img, mask), or None (evaluation)
    norm          : "dataset" | "imagenet" | "none"
    dataset_stats : (mean, std) from the TRAIN partition, used when norm == "dataset"
    test_set      : "official" (1000 images) | "dedup" (899, twins of train images removed)
    include_negatives : segmentation only; add no_tumor images with an all-zero mask
    channels      : 3 replicates gray to RGB for the pretrained encoders
    """

    def __init__(self, X, M, index_df, partition, task="cls", augment=None, norm="dataset",
                 dataset_stats=(0.1686, 0.1838), test_set="official", include_negatives=False, channels=3):
        assert task in ("cls", "seg", "both"), task
        assert norm in ("dataset", "imagenet", "none"), norm
        self.X, self.M, self.task, self.augment, self.norm, self.channels = X, M, task, augment, norm, channels

        df = index_df[index_df["partition"] == partition]
        if partition == "test" and test_set == "dedup":
            df = df[df["test_dedup"]]                                  # drop test images with a twin in train
        if task == "seg" and not include_negatives:
            df = df[df["row_mask"] >= 0]                               # only images that have a mask
        self.cache_rows = df.index.to_numpy()                          # position in the cache_index table
        self.row_img = df["row_img"].to_numpy()                        # position in X
        self.row_mask = df["row_mask"].to_numpy()                      # position in M, -1 = no mask
        self.labels = df["tumor_label"].map(CLASS_TO_IDX).to_numpy()   # 0..3
        self.meta = df.reset_index(drop=True)                          # keep metadata for error analysis later

        if norm == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = (dataset_stats[0],) * 3, (dataset_stats[1],) * 3
        self.mean = torch.tensor(mean[:channels] if channels == 3 else [float(np.mean(mean))]).view(-1, 1, 1)
        self.std = torch.tensor(std[:channels] if channels == 3 else [float(np.mean(std))]).view(-1, 1, 1)

    def __len__(self):
        return len(self.row_img)

    def _normalize(self, img):
        if self.norm == "none":
            return img
        img = img.expand(self.channels, -1, -1)                        # (1,S,S) -> (C,S,S), no copy
        return (img - self.mean) / self.std                            # x_norm = (x - mean) / std

    def __getitem__(self, i):
        img = torch.from_numpy(self.X[self.row_img[i]]).float().div_(255.0).unsqueeze(0)   # (1,S,S) in [0,1]
        need_mask = self.task in ("seg", "both")
        mask = None
        if need_mask:
            rm = self.row_mask[i]
            mask = (torch.from_numpy(self.M[rm]).float().unsqueeze(0) if rm >= 0
                    else torch.zeros_like(img))                        # negatives get an empty mask
        if self.augment is not None:
            img, mask = self.augment(img, mask)                        # joint transform
        out = {"image": self._normalize(img), "index": int(self.cache_rows[i])}
        if self.task in ("cls", "both"):
            out["label"] = int(self.labels[i])
        if need_mask:
            out["mask"] = mask
        return out
