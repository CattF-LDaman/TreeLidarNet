#!/usr/bin/env python3
"""One-shot: after the in-flight epoch is saved, stop that train and resume with the new loss.

The running process loaded the old code. Epoch completion is the checkpoint whose
meta.epoch becomes 2 (persist after the second pass, step ~75500) with a finite epoch_loss.
That file is overwritten ~500 steps later with epoch_loss NaN, so the snapshot is taken
immediately and restored after the old process is killed.
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs/gent_canopy_v2/latest.pt"
LOG = ROOT / "runs/gent_canopy_v2/train_full_grid.log"
ARCHIVE = ROOT / "runs/gent_canopy_v2/archive/pre_per_tree_epoch2"
DONE_EPOCH = 2
POLL_S = 20


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _meta() -> dict | None:
    if not CKPT.is_file():
        return None
    try:
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    except Exception as exc:
        _log(f"checkpoint read failed: {exc}")
        return None
    return ckpt.get("meta") or {}


def _finite_loss(meta: dict) -> float | None:
    try:
        loss_f = float(meta.get("epoch_loss"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(loss_f):
        return None
    return loss_f


def _epoch_saved(meta: dict) -> bool:
    epoch = int(meta.get("epoch", 0))
    step = int(meta.get("global_step", 0))
    return epoch >= DONE_EPOCH and step >= 75000 and _finite_loss(meta) is not None


def _past_window(meta: dict) -> bool:
    """Periodic save already replaced the finite epoch-2 loss (epoch stays 2, loss is NaN)."""
    epoch = int(meta.get("epoch", 0))
    step = int(meta.get("global_step", 0))
    return epoch >= DONE_EPOCH and step >= 76000 and _finite_loss(meta) is None


def _train_pids() -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", "treelidarnet train"], text=True)
    except subprocess.CalledProcessError:
        return []
    own = os.getpgid(os.getpid())
    pids = []
    for line in out.split():
        pid = int(line)
        if pid == os.getpid():
            continue
        try:
            if os.getpgid(pid) == own:
                continue
        except ProcessLookupError:
            continue
        pids.append(pid)
    return pids


def _kill_train() -> None:
    pids = _train_pids()
    if not pids:
        _log("no train process found")
        return
    pgids = set()
    for pid in pids:
        try:
            pgids.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    for pgid in pgids:
        _log(f"kill process group {pgid}")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _ in range(30):
        time.sleep(1)
        if not _train_pids():
            return
    raise RuntimeError("train process still alive after kill")


def _snapshot(note: str) -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "runs/gent_canopy_v2"
    for name in ("latest.pt", "best.pt"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, ARCHIVE / name)
    _log(f"{note}; archived to {ARCHIVE}")


def _restore_snapshot() -> None:
    run_dir = ROOT / "runs/gent_canopy_v2"
    for name in ("latest.pt", "best.pt"):
        src = ARCHIVE / name
        if src.is_file():
            shutil.copy2(src, run_dir / name)
    _log("restored epoch-2 checkpoints over any partial next-epoch save")


def main() -> None:
    os.chdir(ROOT)
    _log("watcher: waiting until epoch 2 checkpoint (meta.epoch>=2); snapshot immediately")
    missed = False
    while True:
        meta = _meta()
        if meta and _epoch_saved(meta):
            _snapshot(
                f"epoch saved epoch={meta.get('epoch')} step={meta.get('global_step')} "
                f"loss={meta.get('epoch_loss')}"
            )
            break
        if meta and _past_window(meta):
            missed = True
            _snapshot(
                f"missed finite-loss window; switching on step={meta.get('global_step')} "
                f"epoch={meta.get('epoch')}"
            )
            break
        time.sleep(POLL_S)

    _kill_train()
    if not missed:
        _restore_snapshot()
    _log("starting train --resume with per-tree loss and fresh optimizer")
    subprocess.Popen(
        ["uv", "run", "treelidarnet", "train", "--config", "configs/gent_canopy_v2.yaml", "--resume"],
        cwd=ROOT,
        stdout=LOG.open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(8)
    _log(f"restarted pids={_train_pids()}")


if __name__ == "__main__":
    main()
