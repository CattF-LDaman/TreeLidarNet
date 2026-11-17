"""Attach per-point RGB from LAZ to existing chips (verifier path; BEV unchanged)."""

from __future__ import annotations

import gc
import multiprocessing as mp
from concurrent.futures import (
    BrokenExecutor,
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.build_gent import index_laz_files, laz_paths_for_chip
from treelidarnet.data.chips import CHIP_SIZE_M
from treelidarnet.data.ground import height_above_ground
from treelidarnet.data.laz_io import read_laz_bbox
from treelidarnet.data.patch_heatmaps import _atomic_save_npz
from treelidarnet.datasets.chip_index import chip_origin_xy
from treelidarnet.paths import chips_dir, data_dir, manifests_dir

_worker_laz_index: dict[tuple[int, int], list[Path]] | None = None
_worker_dtm_path: Path | None = None


def _match_rgb_keys(
    keys_chip: np.ndarray,
    keys_cloud: np.ndarray,
    cloud_rgb: np.ndarray,
    n_chip: int,
    *,
    min_frac: float = 0.9,
) -> np.ndarray | None:
    """Last-wins LUT lookup from cloud RGB onto chip rows."""
    if len(keys_cloud) == 0:
        return None
    lut: dict[tuple[int, ...], int] = {}
    for i in range(len(keys_cloud)):
        lut[tuple(keys_cloud[i].tolist())] = i
    out = np.zeros((n_chip, 3), dtype=np.float32)
    matched = 0
    for i in range(len(keys_chip)):
        j = lut.get(tuple(keys_chip[i].tolist()))
        if j is None:
            continue
        out[i] = cloud_rgb[j]
        matched += 1
    if matched < max(1, int(min_frac * n_chip)):
        return None
    return out


def _match_rgb_to_chip_xyz(
    chip_xyz: np.ndarray,
    cloud_xyz: np.ndarray,
    cloud_rgb: np.ndarray,
    *,
    tol: float = 0.01,
) -> np.ndarray | None:
    """Map RGB from a LAZ cloud onto chip rows via rounded xyz keys."""
    if len(chip_xyz) == len(cloud_xyz) and len(cloud_xyz) > 0:
        return cloud_rgb.astype(np.float32, copy=False)

    if len(cloud_xyz) == 0:
        return None

    keys_cloud_3 = np.round(cloud_xyz / tol).astype(np.int64)
    keys_chip_3 = np.round(chip_xyz / tol).astype(np.int64)
    out = _match_rgb_keys(keys_chip_3, keys_cloud_3, cloud_rgb, len(chip_xyz))
    if out is not None:
        return out

    # Chip z is h_agl at build time; patch recomputes h_agl and z often drifts >1 cm.
    keys_cloud_2 = np.round(cloud_xyz[:, :2] / tol).astype(np.int64)
    keys_chip_2 = np.round(chip_xyz[:, :2] / tol).astype(np.int64)
    return _match_rgb_keys(keys_chip_2, keys_cloud_2, cloud_rgb, len(chip_xyz))


def _chip_origin_from_npz(data: np.lib.npyio.NpzFile, chip_id: str) -> tuple[float, float]:
    if "origin" in data:
        o = data["origin"]
        return float(o[0]), float(o[1])
    parsed = chip_origin_xy(chip_id)
    if parsed is None:
        raise ValueError(f"cannot parse chip origin: {chip_id}")
    return parsed


def _save_chip_with_rgb(chip_path: Path, matched: np.ndarray) -> None:
    with np.load(chip_path, mmap_mode="r") as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    arrays["rgb"] = matched.astype(np.float32, copy=False)
    _atomic_save_npz(chip_path, arrays)
    del arrays


def _chip_has_rgb(data: np.lib.npyio.NpzFile) -> bool:
    if "rgb" not in data.files or "xyz" not in data.files:
        return False
    return len(data["rgb"]) == len(data["xyz"])


def patch_chip_rgb(
    chip_path: Path,
    *,
    laz_index: dict[tuple[int, int], list[Path]],
    dtm_path: Path | None,
    skip_existing: bool = True,
    dedupe_tol: float = 0.01,
) -> bool:
    chip_id = chip_path.stem
    with np.load(chip_path, mmap_mode="r") as data:
        if skip_existing and _chip_has_rgb(data):
            return False
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        if len(xyz) == 0:
            return False
        x0, y0 = _chip_origin_from_npz(data, chip_id)
    x1, y1 = x0 + CHIP_SIZE_M, y0 + CHIP_SIZE_M
    laz_paths = laz_paths_for_chip(x0, y0, laz_index)
    if not laz_paths:
        return False
    global_xyz, feat, rgb = read_laz_bbox(
        laz_paths,
        xmin=x0,
        ymin=y0,
        xmax=x1,
        ymax=y1,
        dedupe_tol=dedupe_tol,
        include_rgb=True,
    )
    if len(global_xyz) == 0:
        del global_xyz, feat, rgb
        return False
    h_agl, _ = height_above_ground(global_xyz, feat, dtm_path=dtm_path)
    local = global_xyz.astype(np.float32, copy=False)
    local[:, 0] -= x0
    local[:, 1] -= y0
    local[:, 2] = h_agl.astype(np.float32)
    del global_xyz, feat, h_agl
    matched = _match_rgb_to_chip_xyz(xyz, local, rgb, tol=dedupe_tol)
    del local, xyz, rgb
    if matched is None:
        return False
    _save_chip_with_rgb(chip_path, matched)
    del matched
    return True


def _worker_init(laz_dir: str, dtm: str | None) -> None:
    global _worker_laz_index, _worker_dtm_path
    _worker_laz_index = index_laz_files(Path(laz_dir))
    _worker_dtm_path = Path(dtm) if dtm else None


def _worker_patch(task: tuple[str, bool, float]) -> tuple[str, bool]:
    chip_path = Path(task[0])
    skip_existing = task[1]
    dedupe_tol = task[2]
    assert _worker_laz_index is not None
    ok = patch_chip_rgb(
        chip_path,
        laz_index=_worker_laz_index,
        dtm_path=_worker_dtm_path,
        skip_existing=skip_existing,
        dedupe_tol=dedupe_tol,
    )
    return chip_path.stem, ok


def _paths_missing_rgb(paths: list[Path], *, scan_every: int = 5000) -> list[Path]:
    """Mmap-only scan; chips that still need an rgb array."""
    need: list[Path] = []
    total = len(paths)
    for i, path in enumerate(paths, 1):
        try:
            with np.load(path, mmap_mode="r") as data:
                if _chip_has_rgb(data):
                    continue
        except OSError:
            continue
        need.append(path)
        if i % scan_every == 0 or i == total:
            print(
                f"scan missing rgb {i}/{total} ({len(need)} need patch)",
                flush=True,
            )
    return need


def _chip_id_list(
    *,
    crops_index: Path | None,
    chips_index: Path | None,
    all_chips: bool,
    register_chips: bool = False,
) -> list[str]:
    if crops_index is not None and crops_index.is_file():
        df = pd.read_parquet(crops_index)
        if register_chips:
            if "source" not in df.columns:
                raise ValueError("register_chips requires source column in crops index")
            df = df[df["source"] == "register"]
        return sorted({str(c) for c in df["chip_id"].astype(str)})
    if chips_index is not None and chips_index.is_file():
        df = pd.read_parquet(chips_index)
        col = "chip_id" if "chip_id" in df.columns else df.columns[0]
        return sorted({str(c) for c in df[col].astype(str)})
    if all_chips:
        full_index = manifests_dir() / "chips_index_full.parquet"
        if full_index.is_file():
            df = pd.read_parquet(full_index)
            return sorted(df["chip_id"].astype(str).tolist())
        return sorted(p.stem for p in chips_dir().glob("*.npz"))
    return sorted(p.stem for p in chips_dir().glob("*.npz"))


def patch_chips_rgb(
    *,
    laz_dir: Path,
    dtm_path: Path | None = None,
    crops_index: Path | None = None,
    chips_index: Path | None = None,
    all_chips: bool = False,
    limit: int | None = None,
    skip_existing: bool = True,
    jobs: int = 1,
    dedupe_tol: float = 0.01,
    offset: int = 0,
    only_missing_rgb: bool = False,
    refresh_missing_manifest: bool = False,
    register_chips: bool = False,
) -> int:
    """Write ``rgb`` (N,3) float32 into chip npz; default targets crops manifest chips only."""
    chip_ids = _chip_id_list(
        crops_index=crops_index,
        chips_index=chips_index,
        all_chips=all_chips,
        register_chips=register_chips,
    )
    if offset > 0:
        chip_ids = chip_ids[offset:]
        print(f"patch rgb: resume offset {offset} → {len(chip_ids)} chip ids left", flush=True)
    if limit:
        chip_ids = chip_ids[:limit]
    if not chip_ids:
        print("Done: rgb on 0 chips (0 requested)", flush=True)
        return 0

    print(f"patch rgb: {len(chip_ids)} chip ids", flush=True)
    chip_paths: list[Path] = []
    for cid in chip_ids:
        path = chips_dir() / f"{cid}.npz"
        if path.is_file():
            chip_paths.append(path)

    manifest_path = missing_rgb_manifest_path(register_chips=register_chips)
    if only_missing_rgb:
        if manifest_path.is_file() and not refresh_missing_manifest:
            ids = pd.read_parquet(manifest_path)["chip_id"].astype(str).tolist()
            chip_paths = [
                chips_dir() / f"{cid}.npz"
                for cid in ids
                if (chips_dir() / f"{cid}.npz").is_file()
            ]
            print(
                f"patch rgb: loaded {len(chip_paths)} chips from {manifest_path.name}",
                flush=True,
            )
        else:
            print(f"patch rgb: scanning for missing rgb...", flush=True)
            chip_paths = _paths_missing_rgb(chip_paths)
            pd.DataFrame({"chip_id": [p.stem for p in chip_paths]}).to_parquet(
                manifest_path, index=False
            )
            print(
                f"patch rgb: {len(chip_paths)} chips need rgb (wrote {manifest_path.name})",
                flush=True,
            )

    patched = 0
    n_tasks = len(chip_paths)
    jobs = max(1, int(jobs))
    print(f"patch rgb: {n_tasks} npz paths, jobs={jobs}, skip_existing={skip_existing}", flush=True)

    def _report(i: int) -> None:
        if i == 1 or i == n_tasks or i % 50 == 0 or i % 500 == 0:
            print(f"patch rgb {i}/{n_tasks} ({patched} written)", flush=True)

    if jobs == 1:
        print(f"patch rgb: indexing LAZ under {laz_dir} ...", flush=True)
        laz_index = index_laz_files(laz_dir)
        if not laz_index:
            raise FileNotFoundError(f"No indexed LAZ in {laz_dir}")
        n_tiles = sum(len(v) for v in laz_index.values())
        print(f"patch rgb: LAZ index ready ({len(laz_index)} corners, {n_tiles} files)", flush=True)
        for i, path in enumerate(chip_paths, 1):
            if patch_chip_rgb(
                path,
                laz_index=laz_index,
                dtm_path=dtm_path,
                skip_existing=skip_existing,
                dedupe_tol=dedupe_tol,
            ):
                patched += 1
            _report(i)
            if i % 10 == 0:
                gc.collect()
    else:
        print(f"patch rgb: parallel workers (re-index LAZ in each child)", flush=True)
        if not index_laz_files(laz_dir):
            raise FileNotFoundError(f"No indexed LAZ in {laz_dir}")
        ctx = mp.get_context("spawn")
        dtm_s = str(dtm_path) if dtm_path else None
        in_flight_max = max(jobs * 2, jobs)
        try:
            with ProcessPoolExecutor(
                max_workers=jobs,
                mp_context=ctx,
                max_tasks_per_child=8,
                initializer=_worker_init,
                initargs=(str(laz_dir), dtm_s),
            ) as pool:
                task_iter = (
                    (str(p), skip_existing, dedupe_tol) for p in chip_paths
                )
                in_flight: set = set()
                done_i = 0
                pending = True

                while in_flight or pending:
                    while len(in_flight) < in_flight_max and pending:
                        try:
                            t = next(task_iter)
                        except StopIteration:
                            pending = False
                            break
                        in_flight.add(pool.submit(_worker_patch, t))
                    if not in_flight:
                        break
                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        in_flight.discard(fut)
                        done_i += 1
                        try:
                            _, ok = fut.result()
                            if ok:
                                patched += 1
                        except Exception as exc:
                            print(f"patch rgb error: {exc}", flush=True)
                        _report(done_i)
        except BrokenExecutor as exc:
            print(
                f"patch rgb: worker pool failed ({exc}) — retry with --jobs 1",
                flush=True,
            )
            raise

    print(f"Done: rgb on {patched} chips ({n_tasks} requested)", flush=True)
    return patched


def default_dtm_path() -> Path | None:
    p = data_dir() / "raw" / "dtm" / "dtm_1m.tif"
    return p if p.is_file() else None


def default_crops_index() -> Path:
    return manifests_dir() / "crops_index.parquet"


def missing_rgb_manifest_path(*, register_chips: bool = False) -> Path:
    if register_chips:
        return manifests_dir() / "chips_register_missing_rgb.parquet"
    return manifests_dir() / "chips_missing_rgb.parquet"