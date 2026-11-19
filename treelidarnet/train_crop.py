"""Train GentCropVerifier on crops_index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from treelidarnet.datasets.crop_dataset import CropDataset, collate_crops, default_crop_manifest
from treelidarnet.models.crop_metrics import crop_val_metrics
from treelidarnet.models.crop_verifier import MODEL_TYPE, GentCropVerifier
from treelidarnet.paths import repo_root, runs_dir


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _manifest_class_counts(manifest: Path, split: str) -> tuple[int, int]:
    df = pd.read_parquet(manifest)
    df = df[df["split"] == split]
    n_pos = int((df["label"] >= 0.5).sum())
    n_neg = int((df["label"] < 0.5).sum())
    return n_pos, n_neg


def save_crop_checkpoint(
    path: Path,
    *,
    model: GentCropVerifier,
    optimizer: torch.optim.Optimizer,
    config_path: Path,
    arch: dict,
    epoch: int,
    global_step: int,
    val_loss: float,
    val_metrics: dict[str, float],
    best_val_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_type": MODEL_TYPE,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "best_val_score": best_val_score,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": meta,
            "arch": arch,
        },
        path,
    )


def _run_val(
    model: GentCropVerifier,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    *,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    val_n = 0
    prob_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in val_loader:
            if batch is None:
                continue
            pts = batch["points"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(pts)
            loss = loss_fn(logits, labels)
            val_loss += float(loss.item())
            val_n += 1
            prob_chunks.append(torch.sigmoid(logits).squeeze(1).cpu())
            label_chunks.append(labels.squeeze(1).cpu())
    val_mean = val_loss / max(val_n, 1) if val_n else float("inf")
    if not prob_chunks:
        return val_mean, {}
    probs = torch.cat(prob_chunks)
    labels = torch.cat(label_chunks)
    metrics = crop_val_metrics(probs, labels, threshold=threshold)
    return val_mean, metrics


def _checkpoint_score(metrics: dict[str, float], val_loss: float) -> float:
    """Higher is better; prefer val F1 when both classes are present."""
    if metrics and "f1" in metrics:
        f1 = float(metrics["f1"])
        if f1 == f1:
            return f1
    return -val_loss


def train_crop_from_config(
    config_path: Path,
    *,
    run_name: str | None = None,
    tensorboard: bool = True,
) -> Path:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_name = run_name or str(cfg.get("run_name", "gent_crop_verifier"))
    run_dir = runs_dir() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(cfg.get("crops_index", default_crop_manifest()))
    if not manifest.is_absolute():
        manifest = repo_root() / manifest

    radius_m = float(cfg.get("radius_m", 8.0))
    num_points = int(cfg.get("num_points", 512))
    min_points = int(cfg.get("min_points", 32))
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 20))
    lr = float(cfg.get("lr", 1e-3))
    num_workers = int(cfg.get("num_workers", 4))
    val_threshold = float(cfg.get("val_threshold", 0.5))
    balance_class_loss = bool(cfg.get("balance_class_loss", True))
    explorer_fp_loss_weight = float(cfg.get("explorer_fp_loss_weight", 1.0))
    explorer_fp_sample_weight = float(cfg.get("explorer_fp_sample_weight", 1.0))
    resume_from = cfg.get("resume_from")
    resume_path = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = repo_root() / resume_path

    arch = {
        "in_channels": int(cfg.get("in_channels", 6)),
        "sa1_npoint": int(cfg.get("sa1_npoint", 128)),
        "sa1_radius": float(cfg.get("sa1_radius", 0.5)),
        "sa1_nsample": int(cfg.get("sa1_nsample", 32)),
        "sa2_npoint": int(cfg.get("sa2_npoint", 32)),
        "sa2_radius": float(cfg.get("sa2_radius", 2.0)),
        "sa2_nsample": int(cfg.get("sa2_nsample", 64)),
    }

    in_channels = int(cfg.get("in_channels", 6))
    train_ds = CropDataset(
        manifest,
        split="train",
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        in_channels=in_channels,
        explorer_fp_loss_weight=explorer_fp_loss_weight,
        explorer_fp_sample_weight=explorer_fp_sample_weight,
    )
    val_ds = CropDataset(
        manifest,
        split="val",
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        seed=0,
        in_channels=in_channels,
        explorer_fp_loss_weight=1.0,
        explorer_fp_sample_weight=1.0,
    )
    if len(val_ds) == 0:
        val_ds = CropDataset(
            manifest,
            split="test",
            radius_m=radius_m,
            num_points=num_points,
            min_points=min_points,
            seed=0,
            in_channels=in_channels,
        )

    n_pos, n_neg = _manifest_class_counts(manifest, "train")
    print(f"Train manifest: {n_pos} pos, {n_neg} neg (balance_class_loss={balance_class_loss})", flush=True)
    val_pos, val_neg = _manifest_class_counts(manifest, "val")
    print(f"Val manifest: {val_pos} pos, {val_neg} neg", flush=True)

    train_df = pd.read_parquet(manifest)
    train_df = train_df[train_df["split"] == "train"]
    n_fp_train = int((train_df.get("source", pd.Series(dtype=str)) == "explorer_fp").sum())
    if n_fp_train and (
        explorer_fp_loss_weight > 1.0 or explorer_fp_sample_weight > 1.0
    ):
        print(
            f"explorer_fp train rows: {n_fp_train} "
            f"(loss_weight={explorer_fp_loss_weight}, sample_weight={explorer_fp_sample_weight})",
            flush=True,
        )

    train_sampler = None
    shuffle = True
    if explorer_fp_sample_weight > 1.0 and len(train_ds) > 0:
        sw = train_ds.sample_weights()
        train_sampler = WeightedRandomSampler(
            sw, num_samples=len(sw), replacement=True
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_crops,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_crops,
        pin_memory=torch.cuda.is_available(),
    )

    device = _device()
    model = GentCropVerifier(**arch).to(device)
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from not found: {resume_path}")
        from treelidarnet.models.crop_verifier import load_crop_verifier

        ckpt_model = load_crop_verifier(resume_path, device)
        model.load_state_dict(ckpt_model.state_dict())
        print(f"Resumed model weights from {resume_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_weight = None
    if balance_class_loss and n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        print(f"BCE pos_weight={pos_weight.item():.4f} (upweights rare class in the loss)", flush=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_row_weights = explorer_fp_loss_weight > 1.0

    writer = SummaryWriter(run_dir / "tb") if tensorboard else None
    if writer:
        tb_dir = run_dir / "tb"
        print(f"TensorBoard logdir: {tb_dir}", flush=True)
        print(f"  http://127.0.0.1:6006/  (logdir: {runs_dir()})", flush=True)

    best_val_score = float("-inf")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            pts = batch["points"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pts)
            if use_row_weights:
                per = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight, reduction="none"
                )
                w = batch["loss_weight"].to(device).unsqueeze(1)
                loss = (per * w).sum() / w.sum().clamp_min(1e-8)
            else:
                loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            if writer and global_step % 20 == 0:
                writer.add_scalar("loss/train_step", loss.item(), global_step)

        train_mean = train_loss / max(n_batches, 1)
        val_mean, val_metrics = _run_val(
            model, val_loader, device, loss_fn, threshold=val_threshold
        )

        metric_str = ""
        if val_metrics:
            metric_str = (
                f" val_f1={val_metrics['f1']:.3f}"
                f" prec={val_metrics['precision']:.3f}"
                f" rec={val_metrics['recall']:.3f}"
            )
            if "auroc" in val_metrics:
                metric_str += f" auroc={val_metrics['auroc']:.3f}"
            metric_str += (
                f" mean_p_pos={val_metrics.get('mean_prob_pos', float('nan')):.3f}"
                f" mean_p_neg={val_metrics.get('mean_prob_neg', float('nan')):.3f}"
            )

        print(
            f"epoch {epoch + 1}/{epochs} train_loss={train_mean:.4f} val_loss={val_mean:.4f}{metric_str}",
            flush=True,
        )
        if writer:
            writer.add_scalar("loss/train_epoch", train_mean, epoch)
            writer.add_scalar("loss/val_epoch", val_mean, epoch)
            for key, value in val_metrics.items():
                if value == value:
                    writer.add_scalar(f"val/{key}", value, epoch)

        val_score = _checkpoint_score(val_metrics, val_mean)

        save_crop_checkpoint(
            run_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            config_path=config_path,
            arch=arch,
            epoch=epoch + 1,
            global_step=global_step,
            val_loss=val_mean,
            val_metrics=val_metrics,
            best_val_score=best_val_score,
        )
        if val_score > best_val_score:
            best_val_score = val_score
            save_crop_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config_path=config_path,
                arch=arch,
                epoch=epoch + 1,
                global_step=global_step,
                val_loss=val_mean,
                val_metrics=val_metrics,
                best_val_score=best_val_score,
            )

    if writer:
        writer.close()
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Done. best val score={best_val_score:.4f} -> {run_dir / 'best.pt'}", flush=True)
    return run_dir / "best.pt"