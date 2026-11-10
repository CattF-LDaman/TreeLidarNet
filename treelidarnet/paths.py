"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _REPO_ROOT


def data_dir() -> Path:
    return _REPO_ROOT / "data"


def manifests_dir() -> Path:
    d = data_dir() / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir() -> Path:
    d = data_dir() / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def chips_dir() -> Path:
    d = processed_dir() / "chips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = _REPO_ROOT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exports_dir() -> Path:
    d = _REPO_ROOT / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d
