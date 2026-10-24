"""Staged fine-tuning engine: head-only warm-up, then full fine-tuning with warm-up + cosine LR and early stopping."""
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .evaluate import predict, metrics_from_logits


def set_backbone_trainable(model, trainable):
    """Freeze/unfreeze all feature layers. requires_grad=False => no gradient, no update."""
    for p in model.features.parameters():
        p.requires_grad = trainable


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    """AdamW (Loshchilov & Hutter, 2019): Adam with DECOUPLED weight decay  w <- w - lr * wd * w.
    Two learning rates: small for pretrained backbone, larger for the fresh head.
    Biases and BatchNorm parameters (ndim <= 1) get no weight decay: shrinking them hurts, helps nothing."""
    groups = []
    for params, lr in [(model.features.parameters(), lr_backbone), (model.classifier.parameters(), lr_head)]:
        decay, no_decay = [], []
        for p in params:
            if p.requires_grad:                       # frozen layers are skipped (phase 1)
                (no_decay if p.ndim <= 1 else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def make_scheduler(opt, total_steps, warmup_steps, min_factor=0.01):
    """Per-step schedule: linear warm-up, then cosine decay (Loshchilov & Hutter, 2017):
        factor(t) = t/W                                              for t < W
        factor(t) = m + (1-m) * 0.5 * (1 + cos(pi * (t-W)/(T-W)))    afterwards
    The LR of every group is its base LR times factor(t). Warm-up avoids large early updates when
    the backbone first unfreezes; the decay lets training settle into a minimum."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=f)


def train_one_epoch(model, loader, opt, sched, scaler, loss_fn, device, backbone_frozen, clip):
    model.train()
    if backbone_frozen:
        model.features.eval()      # keep BatchNorm running statistics fixed while the backbone is frozen
    tot_loss = torch.zeros((), device=device)
    tot_correct = torch.zeros((), device=device)
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):       # mixed precision forward
            logits = model(x)
        loss = loss_fn(logits.float(), y)                        # loss in float32 for numerical safety
        scaler.scale(loss).backward()                            # scaled loss => float16 gradients do not underflow
        scaler.unscale_(opt)                                     # back to true gradient scale ...
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # ... so that clipping uses real norms
        scaler.step(opt)                                         # skips the update if gradients are inf/nan
        scaler.update()
        if sched is not None:
            sched.step()                                         # (a harmless warning may appear on step 1)
        tot_loss += loss.detach() * len(y)
        tot_correct += (logits.argmax(1) == y).sum()
        n += len(y)
    return float(tot_loss) / n, float(tot_correct) / n


def fit(model, loaders, cfg, run_name, device, class_names, models_dir, logs_dir, tables_dir):
    models_dir, logs_dir, tables_dir = Path(models_dir), Path(logs_dir), Path(tables_dir)
    writer = SummaryWriter(log_dir=str(logs_dir / run_name))      # TensorBoard event files
    best_path = models_dir / f"{run_name}_best.pt"
    scaler = torch.amp.GradScaler("cuda")
    w = cfg.get("class_weights")
    loss_fn = nn.CrossEntropyLoss(
        weight=None if w is None else torch.tensor(w, dtype=torch.float32, device=device),
        label_smoothing=cfg.get("label_smoothing", 0.0))          # CE = -log softmax(z)[y]
    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "bad": 0}
    history = []

    def run_phase(phase, n_epochs, opt, sched, frozen, early_stop):
        for _ in range(n_epochs):
            state["epoch"] += 1
            ep = state["epoch"]
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_one_epoch(model, loaders["train"], opt, sched, scaler, loss_fn, device, frozen, cfg["clip"])
            logits, labels, _ = predict(model, loaders["val"], device)
            m = metrics_from_logits(logits, labels, class_names)
            row = {"epoch": ep, "phase": phase, "train_loss": tr_loss, "train_acc": tr_acc,
                   "val_nll": m["nll"], "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                   "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t0}
            history.append(row)
            for k in ["train_loss", "train_acc", "val_nll", "val_acc", "val_macro_f1", "lr"]:
                writer.add_scalar(k, row[k], ep)
            if m["macro_f1"] > state["best_f1"] + cfg["min_delta"]:        # a real improvement
                state.update(best_f1=m["macro_f1"], best_epoch=ep, bad=0)
                torch.save({"model": model.state_dict(), "epoch": ep, "val_macro_f1": float(m["macro_f1"]),
                            "cfg": cfg, "class_names": list(class_names)}, best_path)   # checkpoint of the best model
            elif early_stop:
                state["bad"] += 1
            print(f"ep {ep:2d} [{phase:8s}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val nll {m['nll']:.4f} "
                  f"acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} | best {state['best_f1']:.4f}@{state['best_epoch']} "
                  f"| {row['seconds']:.0f}s")
            if early_stop and state["bad"] >= cfg["patience"]:
                print(f"early stopping: no improvement for {cfg['patience']} epochs")
                return

    # ---- phase 1: head only (backbone frozen), constant LR ----
    set_backbone_trainable(model, False)
    opt = make_optimizer(model, cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"])
    if cfg["epochs_head"] > 0:
        run_phase("head", cfg["epochs_head"], opt, None, True, False)

    # ---- phase 2: everything trainable, warm-up + cosine, early stopping on val macro-F1 ----
    set_backbone_trainable(model, True)
    opt = make_optimizer(model, cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"])
    steps = len(loaders["train"])
    sched = make_scheduler(opt, cfg["epochs_ft"] * steps, int(cfg["warmup_epochs"] * steps))
    run_phase("finetune", cfg["epochs_ft"], opt, sched, False, True)

    writer.close()
    ckpt = torch.load(best_path, map_location=device)             # reload the BEST weights, not the last
    model.load_state_dict(ckpt["model"])
    hist = pd.DataFrame(history)
    hist.to_csv(tables_dir / f"{run_name}_history.csv", index=False)
    return hist, state
