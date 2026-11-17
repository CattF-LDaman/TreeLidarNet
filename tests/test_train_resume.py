from pathlib import Path

from treelidarnet.train import (
    archive_checkpoint_once,
    resume_checkpoint_path,
    resume_epoch_stop,
)


def test_resume_from_best_ignores_latest(tmp_path: Path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    latest.write_bytes(b"latest")
    best.write_bytes(b"best")
    assert resume_checkpoint_path("best", latest=latest, best=best) == best
    assert resume_checkpoint_path("latest", latest=latest, best=best) == latest


def test_resume_epoch_stop_is_one_past_the_checkpoint():
    assert resume_epoch_stop(3, 10, 1) == 4
    assert resume_epoch_stop(3, 10, None) == 10


def test_archive_checkpoint_once_does_not_overwrite(tmp_path: Path):
    src = tmp_path / "best.pt"
    dest = tmp_path / "archive" / "epoch3_hot" / "best.pt"
    src.write_bytes(b"epoch3")
    assert archive_checkpoint_once(src, dest) == dest
    src.write_bytes(b"newer")
    assert archive_checkpoint_once(src, dest) is None
    assert dest.read_bytes() == b"epoch3"
