"""Two-phase segmentation training with a switchable encoder policy.
  phase 1 (decoder warm-up): encoder frozen.
  phase 2: warm-up + cosine LR; encoder policy from cfg["enc_mode"]:
      "full"    all encoder layers trainable
      "partial" only the deepest MBConv stage (features[7], stride 32) trainable (the Diagnostics 2026 strategy)
      "frozen"  encoder frozen; decoder and head only
Frozen layers ALSO keep BatchNorm in eval mode: otherwise their running mean/variance would still drift on new data."""
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from src.classification.train import make_scheduler
from .losses import SegLoss
from .metrics import quick_eval


def set_encoder_mode(model, mode):
    assert mode in ("frozen", "partial", "full"), mode
    for p in model.encoder.parameters():
        p.requires_grad = (mode == "full")
    if mode == "partial":
        for p in model.encoder.features[7].parameters():      # deepest stage: stride 32, 320 channels
            p.requires_grad = True


def set_encoder_trainable(model, trainable):                  # backward-compatible alias
    set_encoder_mode(model, "full" if trainable else "frozen")


def set_train_mode(model, mode, warmup_phase):
    """model.train() switches EVERYTHING to training behaviour; then re-freeze the BatchNorm of frozen encoder parts."""
    model.train()
    if warmup_phase or mode == "frozen":
        model.encoder.eval()
    elif mode == "partial":
        model.encoder.eval()
        model.encoder.features[7].train()


def make_optimizer(model, lr_enc, lr_dec, weight_decay):
    """AdamW; separate LRs; no weight decay on biases / BatchNorm (ndim <= 1). Frozen parameters are skipped."""
    groups = []
    for params, lr in [(model.encoder.parameters(), lr_enc), (model.decoder_parameters(), lr_dec)]:
        decay, no_decay = [], []
        for p in params:
            if p.requires_grad:
                (no_decay if p.ndim <= 1 else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def train_epoch(model, loader, opt, sched, scaler, loss_fn, device, enc_mode, warmup_phase, clip):
    set_train_mode(model, enc_mode, warmup_phase)
    tot, n = torch.zeros((), device=device), 0
    for b in loader:
        x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = loss_fn(logits.float(), y)                     # loss in float32
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt); scaler.update()
        if sched is not None:
            sched.step()
        tot += loss.detach() * len(x); n += len(x)
    return float(tot) / n


def fit_seg(model, loaders, cfg, run_name, device, models_dir, logs_dir, tables_dir):
    models_dir, logs_dir, tables_dir = Path(models_dir), Path(logs_dir), Path(tables_dir)
    writer = SummaryWriter(log_dir=str(logs_dir / run_name))
    best_path = models_dir / f"{run_name}_best.pt"
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = SegLoss(cfg["loss"])
    state, history = {"epoch": 0, "best": -1.0, "best_epoch": 0, "bad": 0}, []

    def run_phase(phase, n_epochs, opt, sched, mode, warmup_phase, early_stop):
        for _ in range(n_epochs):
            state["epoch"] += 1
            ep, t0 = state["epoch"], time.perf_counter()
            tr = train_epoch(model, loaders["train"], opt, sched, scaler, loss_fn, device, mode, warmup_phase, cfg["clip"])
            vl, vd, vi = quick_eval(model, loaders["val"], device, loss_fn)
            row = {"epoch": ep, "phase": phase, "train_loss": tr, "val_loss": vl, "val_dice": vd, "val_iou": vi,
                   "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t0}
            history.append(row)
            for k in ("train_loss", "val_loss", "val_dice", "val_iou", "lr"):
                writer.add_scalar(k, row[k], ep)
            if vd > state["best"] + cfg["min_delta"]:
                state.update(best=vd, best_epoch=ep, bad=0)
                torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": float(vd), "cfg": cfg}, best_path)
            elif early_stop:
                state["bad"] += 1
            print(f"ep {ep:2d} [{phase:8s}] train {tr:.4f} | val loss {vl:.4f} dice {vd:.4f} iou {vi:.4f} "
                  f"| best {state['best']:.4f}@{state['best_epoch']} | {row['seconds']:.0f}s")
            if early_stop and state["bad"] >= cfg["patience"]:
                print(f"early stopping: no improvement for {cfg['patience']} epochs")
                return

    mode = cfg.get("enc_mode", "full")
    set_encoder_mode(model, "frozen")                          # phase 1: decoder warm-up
    opt = make_optimizer(model, cfg["lr_enc"], cfg["lr_dec"], cfg["weight_decay"])
    if cfg["epochs_dec"] > 0:
        run_phase("decoder", cfg["epochs_dec"], opt, None, "frozen", True, False)
    set_encoder_mode(model, mode)                              # phase 2: chosen encoder policy
    opt = make_optimizer(model, cfg["lr_enc"], cfg["lr_dec"], cfg["weight_decay"])
    steps = len(loaders["train"])
    sched = make_scheduler(opt, cfg["epochs_ft"] * steps, int(cfg["warmup_epochs"] * steps))
    run_phase("finetune", cfg["epochs_ft"], opt, sched, mode, False, True)

    writer.close()
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])    # reload the BEST weights
    hist = pd.DataFrame(history)
    hist.to_csv(tables_dir / f"{run_name}_history.csv", index=False)
    return hist, state
