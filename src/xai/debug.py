import time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from src.xai.methods import MultiGradCAM, to_input_batch, energy_batch
from src.classification import train as TR
from src.classification.evaluate import predict, metrics_from_logits
from src.preprocessing.dataset import CLASS_ORDER, CLASS_TO_IDX
NO_TUMOR = CLASS_ORDER.index("no_tumor")


def head_masks(imgs_u8, thr=12):
    """Rough head region: pixels above `thr`, closed and hole-filled. Background (incl. padding bars) = outside it."""
    return np.stack([ndimage.binary_fill_holes(ndimage.binary_closing(im > thr, iterations=3)) for im in imgs_u8])


def attention_table(model, S, rows, norm, layer=8, bs=64):
    """Grad-CAM (features[layer]) of the PREDICTED class for the given cache rows: energy inside the GT mask and on background."""
    mc = MultiGradCAM(model, {"L": model.features[layer]}); rec = []
    for i0 in range(0, len(rows), bs):
        r = np.asarray(rows[i0:i0 + bs]); imgs = S.Xc[r]
        cams, logits, tgt = mc(to_input_batch(imgs, norm, S.STATS, S.DEVICE))
        cam = cams["L"].cpu().numpy(); prob = torch.softmax(logits.float(), 1)
        rm = S.cache_index.loc[r, "row_mask"].values; has = rm >= 0
        gt = np.zeros(imgs.shape, bool)
        if has.any(): gt[has] = S.Mc[rm[has]] > 0
        e_in = energy_batch(cam, gt); e_in[~has] = np.nan
        e_bg = energy_batch(cam, ~head_masks(imgs))
        true = S.cache_index.loc[r, "tumor_label"].map(CLASS_TO_IDX).values
        for j in range(len(r)):
            rec.append((int(r[j]), int(true[j]), int(tgt[j]), float(prob[j, tgt[j]]), float(e_in[j]), float(e_bg[j])))
    mc.remove()
    return pd.DataFrame(rec, columns=["idx", "true", "pred", "conf", "e_in", "e_bg"])


def train_epoch_attn(model, loader, opt, sched, scaler, device, frozen, clip, lam):
    model.train()
    if frozen: model.features.eval()
    tot = torch.zeros((), device=device); n = 0
    for b in loader:
        x, y, m = b["image"].to(device), b["label"].to(device), b["mask"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = model.features(x)                                        # (B,1280,8,8)
            logits = model.classifier(model.avgpool(feats).flatten(1))      # identical to model(x)
        loss = F.cross_entropy(logits.float(), y)
        if lam > 0:
            W = model.classifier[1].weight                                  # (K,1280)
            cam = F.relu(torch.einsum("bkhw,bk->bhw", feats.float(), W[y].float()))   # CAM of the TRUE class (= Grad-CAM for GAP+linear)
            mfrac = F.adaptive_avg_pool2d(m, feats.shape[-2:])[:, 0]        # tumor fraction in each 8x8 cell
            has = (y != NO_TUMOR) & (m.flatten(1).sum(1) > 0)
            if has.any():
                inside = (cam * mfrac).sum((1, 2)); total = cam.sum((1, 2)) + 1e-6
                loss = loss + lam * (1 - inside / total)[has].mean()        # penalise attention mass outside the tumor
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt); scaler.update()
        if sched is not None: sched.step()
        tot += loss.detach() * len(y); n += len(y)
    return float(tot) / n


def make_train_loader(ds, weights, bs, nw, seed):
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=nw, pin_memory=True, drop_last=True, generator=g)
    if nw > 0: kw.update(persistent_workers=True, prefetch_factor=2)
    if weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(ds, batch_size=bs, sampler=sampler, **kw)
    return DataLoader(ds, batch_size=bs, shuffle=True, **kw)


def fit_attn(model, loaders, cfg, run_name, device, models_dir, tables_dir):
    """Same schedule as src.classification.train.fit (head warm-up, then warm-up + cosine, early stopping on val macro-F1)."""
    best_path = Path(models_dir) / f"{run_name}_best.pt"; scaler = torch.amp.GradScaler("cuda")
    state, hist = {"epoch": 0, "best": -1.0, "bad": 0, "best_epoch": 0}, []

    def phase(name, n_ep, opt, sched, frozen, early):
        for _ in range(n_ep):
            state["epoch"] += 1; ep = state["epoch"]; t0 = time.perf_counter()
            tr = train_epoch_attn(model, loaders["train"], opt, sched, scaler, device, frozen, cfg["clip"], cfg["lam"])
            lg, lb, _ = predict(model, loaders["val"], device); m = metrics_from_logits(lg, lb, CLASS_ORDER)
            hist.append(dict(epoch=ep, phase=name, train_loss=tr, val_macro_f1=m["macro_f1"], val_acc=m["accuracy"], sec=time.perf_counter() - t0))
            if m["macro_f1"] > state["best"] + cfg["min_delta"]:
                state.update(best=m["macro_f1"], best_epoch=ep, bad=0)
                torch.save({"model": model.state_dict(), "epoch": ep, "val_macro_f1": float(m["macro_f1"]), "cfg": cfg,
                            "class_names": list(CLASS_ORDER)}, best_path)
            elif early: state["bad"] += 1
            print(f"ep {ep:2d} [{name:8s}] loss {tr:.4f} | val F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} | best {state['best']:.4f}@{state['best_epoch']}")
            if early and state["bad"] >= cfg["patience"]: return

    TR.set_backbone_trainable(model, False)
    opt = TR.make_optimizer(model, cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"])
    phase("head", cfg["epochs_head"], opt, None, True, False)
    TR.set_backbone_trainable(model, True)
    opt = TR.make_optimizer(model, cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"])
    steps = len(loaders["train"]); sched = TR.make_scheduler(opt, cfg["epochs_ft"] * steps, int(cfg["warmup_epochs"] * steps))
    phase("finetune", cfg["epochs_ft"], opt, sched, False, True)
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    pd.DataFrame(hist).to_csv(Path(tables_dir) / f"{run_name}_history.csv", index=False)
    return state
