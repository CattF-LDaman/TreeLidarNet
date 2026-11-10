"""Train GentTreeDetector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from treelidarnet.datasets.chip_dataset import ChipDataset, collate_chips
from treelidarnet.models.gent_detector import GentTreeDetector
from treelidarnet.models.losses import focal_loss_heatmap
from treelidarnet.paths import chips_dir, manifests_dir, runs_dir


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_meta(config_path: Path) -> dict:
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    split_meta = manifests_dir() / "split_meta.json"
    if split_meta.is_file():
        meta["split_meta"] = json.loads(split_meta.read_text(encoding="utf-8"))
    return meta


def save_checkpoint(
    path: Path,
    *,
    model: GentTreeDetector,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config_path: Path,
    in_channels: int,
    epoch: int,
    global_step: int,
    epoch_loss: float,
    best_epoch_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = _run_meta(config_path)
    meta.update(
        {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_loss": epoch_loss,
            "best_epoch_loss": best_epoch_loss,
        }
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "meta": meta,
            "in_channels": in_channels,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: GentTreeDetector,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    """Return (start_epoch, global_step, best_epoch_loss); meta.epoch is next epoch (0-based)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    meta = ckpt.get("meta") or {}
    start_epoch = int(meta.get("epoch", 0))
    global_step = int(meta.get("global_step", 0))
    best = float(meta.get("best_epoch_loss", meta.get("epoch_loss", float("inf"))))
    return start_epoch, global_step, best


def train_from_config(
    config_path: Path,
    *,
    run_name: str | None = None,
    tensorboard: bool = True,
    resume: bool = False,
) -> Path:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chip_glob = cfg.get("chip_glob", "*.npz")
    paths = sorted(chips_dir().glob(chip_glob))
    if not paths:
        raise FileNotFoundError(f"No chips matching {chip_glob} in {chips_dir()}")
    ds = ChipDataset(paths)
    in_ch = int(cfg.get("in_channels", 4))
    batch = int(cfg.get("batch_size", 2))
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-3))
    checkpoint_every_steps = int(cfg.get("checkpoint_every_steps", 500))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_chips)
    device = _device()
    model = GentTreeDetector(in_channels=in_ch, base=int(cfg.get("base_channels", 32))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    run_name = run_name or cfg.get("run_name", "gent_bev_v1")
    out_dir = runs_dir() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"

    start_epoch = 0
    global_step = 0
    best_epoch_loss = float("inf")
    if resume and latest_path.is_file():
        start_epoch, global_step, best_epoch_loss = load_training_checkpoint(
            latest_path, model=model, optimizer=opt, scaler=scaler, device=device
        )
        print(f"Resumed from {latest_path} (epoch index {start_epoch}, step {global_step})")
    elif resume and best_path.is_file():
        start_epoch, global_step, best_epoch_loss = load_training_checkpoint(
            best_path, model=model, optimizer=opt, scaler=scaler, device=device
        )
        print(f"Resumed from {best_path} (epoch index {start_epoch}, step {global_step})")

    writer: SummaryWriter | None = None
    if tensorboard:
        tb_dir = out_dir / "tb"
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard logdir: {tb_dir}")
        print(f"  uv run tensorboard --logdir {out_dir.parent} --bind_all")

    def persist(next_epoch: int, epoch_loss: float) -> None:
        nonlocal best_epoch_loss
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=opt,
            scaler=scaler,
            config_path=config_path,
            in_channels=in_ch,
            epoch=next_epoch,
            global_step=global_step,
            epoch_loss=epoch_loss,
            best_epoch_loss=best_epoch_loss,
        )
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            save_checkpoint(
                best_path,
                model=model,
                optimizer=opt,
                scaler=scaler,
                config_path=config_path,
                in_channels=in_ch,
                epoch=next_epoch,
                global_step=global_step,
                epoch_loss=epoch_loss,
                best_epoch_loss=best_epoch_loss,
            )
            print(f"  saved best.pt (epoch loss {epoch_loss:.4f})")

    resume_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs):
            resume_epoch = epoch
            model.train()
            total = 0.0
            last_pred = None
            last_target = None
            for batch_t in loader:
                bev = batch_t["bev"].to(device)
                target = batch_t["heatmap"].to(device)
                ignore = batch_t["ignore"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(bev)
                    loss = focal_loss_heatmap(pred, target, ignore)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_val = float(loss.item())
                if not (loss_val == loss_val and loss_val < 1e6):
                    msg = (
                        f"Training stopped: bad loss {loss_val}"
                        f" at epoch {epoch + 1} step {global_step + 1}"
                    )
                    if writer:
                        writer.close()
                    raise RuntimeError(msg)
                total += loss_val
                global_step += 1
                if writer:
                    writer.add_scalar("loss/train_step", loss_val, global_step)
                last_pred = pred.detach().cpu()
                last_target = target.detach().cpu()
                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    save_checkpoint(
                        latest_path,
                        model=model,
                        optimizer=opt,
                        scaler=scaler,
                        config_path=config_path,
                        in_channels=in_ch,
                        epoch=epoch,
                        global_step=global_step,
                        epoch_loss=float("nan"),
                        best_epoch_loss=best_epoch_loss,
                    )
                    print(f"  checkpoint latest.pt @ step {global_step}")
            avg = total / max(len(loader), 1)
            current_lr = float(opt.param_groups[0]["lr"])
            if writer:
                writer.add_scalar("loss/epoch", avg, epoch + 1)
                writer.add_scalar("lr", current_lr, epoch + 1)
                if last_pred is not None:
                    writer.add_images("heatmap/pred", last_pred[:4], epoch + 1)
                    writer.add_images("heatmap/target", last_target[:4], epoch + 1)
            print(f"epoch {epoch + 1}/{epochs} loss={avg:.4f}")
            persist(epoch + 1, avg)
    except KeyboardInterrupt:
        print("Interrupted — writing latest.pt")
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=opt,
            scaler=scaler,
            config_path=config_path,
            in_channels=in_ch,
            epoch=resume_epoch,
            global_step=global_step,
            epoch_loss=float("nan"),
            best_epoch_loss=best_epoch_loss,
        )
        if writer:
            writer.flush()
            writer.close()
        raise

    if writer:
        writer.flush()
        writer.close()

    if not best_path.is_file():
        persist(epochs, float("nan"))
    return best_path if best_path.is_file() else latest_path
