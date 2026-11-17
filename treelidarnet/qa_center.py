"""Center-chip QA: compare two checkpoints before promoting best.pt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from treelidarnet.eval import load_ground_truths
from treelidarnet.infer_city import infer_gent_city
from treelidarnet.paths import exports_dir, manifests_dir, repo_root
from treelidarnet.postprocess import Detection


@dataclass
class CenterQAResult:
    chip_count: int
    baseline_rings_per_chip: float
    candidate_rings_per_chip: float
    high_conf_baseline: int
    high_conf_kept: int
    high_conf_recall: float
    passed: bool
    notes: str


def _preds_at_threshold(preds: list[Detection], thresh: float) -> list[Detection]:
    return [p for p in preds if p.score >= thresh]


def _best_score_near(
    preds: list[Detection], gx: float, gy: float, *, radius_m: float
) -> float:
    r2 = radius_m * radius_m
    best = 0.0
    for p in preds:
        dx = p.x - gx
        dy = p.y - gy
        if dx * dx + dy * dy <= r2:
            best = max(best, p.score)
    return best


def compare_center_checkpoints(
    *,
    baseline_ckpt: Path,
    candidate_ckpt: Path,
    trees_parquet: Path | None = None,
    chips_index: Path | None = None,
    limit: int = 400,
    export_floor: float = 0.20,
    high_thresh: float = 0.90,
    match_m: float = 3.0,
    min_high_recall: float = 0.98,
) -> CenterQAResult:
    """Infer the first ``limit`` center-ordered chips for both checkpoints."""
    trees_parquet = trees_parquet or (manifests_dir() / "gent_trees.parquet")
    chips_index = chips_index or (manifests_dir() / "chips_index_full.parquet")
    qa_dir = exports_dir() / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    baseline_export = qa_dir / "center_baseline.json.gz"
    candidate_export = qa_dir / "center_candidate.json.gz"

    infer_gent_city(
        checkpoint=baseline_ckpt,
        trees_parquet=trees_parquet,
        export_path=baseline_export,
        model_id="qa_baseline",
        limit=limit,
        chips_index=chips_index,
        score_threshold=export_floor,
        live=False,
    )
    infer_gent_city(
        checkpoint=candidate_ckpt,
        trees_parquet=trees_parquet,
        export_path=candidate_export,
        model_id="qa_candidate",
        limit=limit,
        chips_index=chips_index,
        score_threshold=export_floor,
        live=False,
    )

    baseline_preds = _load_export_lambert(baseline_export)
    candidate_preds = _load_export_lambert(candidate_export)
    base_f = _preds_at_threshold(baseline_preds, export_floor)
    cand_f = _preds_at_threshold(candidate_preds, export_floor)
    baseline_rpc = len(base_f) / max(limit, 1)
    candidate_rpc = len(cand_f) / max(limit, 1)

    gts = load_ground_truths(trees_parquet, eligible_only=True)
    high_baseline = 0
    high_kept = 0
    for gt in gts:
        b = _best_score_near(baseline_preds, gt.x, gt.y, radius_m=match_m)
        if b < high_thresh:
            continue
        high_baseline += 1
        c = _best_score_near(candidate_preds, gt.x, gt.y, radius_m=match_m)
        if c >= high_thresh:
            high_kept += 1
    recall = high_kept / high_baseline if high_baseline else 1.0

    rings_ok = candidate_rpc < baseline_rpc
    recall_ok = recall >= min_high_recall
    passed = rings_ok and recall_ok
    notes = (
        f"rings/chip {candidate_rpc:.2f} vs baseline {baseline_rpc:.2f}; "
        f"high-conf recall {high_kept}/{high_baseline} ({100 * recall:.1f}%)"
    )
    return CenterQAResult(
        chip_count=limit,
        baseline_rings_per_chip=baseline_rpc,
        candidate_rings_per_chip=candidate_rpc,
        high_conf_baseline=high_baseline,
        high_conf_kept=high_kept,
        high_conf_recall=recall,
        passed=passed,
        notes=notes,
    )


def _load_export_lambert(path: Path) -> list[Detection]:
    import gzip

    from treelidarnet.data.crs import wgs84_to_lambert

    with gzip.open(path, "rb") as handle:
        payload = json.loads(handle.read())
    out: list[Detection] = []
    for row in payload.get("predictions", []):
        lon, lat = row["position"]
        x, y = wgs84_to_lambert(lon, lat)
        out.append(Detection(x=x, y=y, score=float(row.get("score", 0))))
    return out


def promote_or_revert(
    *,
    result: CenterQAResult,
    baseline_ckpt: Path,
    candidate_ckpt: Path,
    live_best: Path | None = None,
    failed_archive: Path | None = None,
) -> bool:
    """If QA failed, copy baseline back to runs/.../best.pt and archive candidate."""
    live_best = live_best or (repo_root() / "runs/gent_canopy_v2/best.pt")
    failed_archive = failed_archive or (
        repo_root() / "runs/gent_canopy_v2/archive/epoch4b_qa_fail/best.pt"
    )
    if result.passed:
        print(f"QA passed — keeping {candidate_ckpt} as {live_best}", flush=True)
        return True
    failed_archive.parent.mkdir(parents=True, exist_ok=True)
    if candidate_ckpt.resolve() != failed_archive.resolve():
        import shutil

        shutil.copy2(candidate_ckpt, failed_archive)
    import shutil

    shutil.copy2(baseline_ckpt, live_best)
    print(
        f"QA failed — restored {live_best} from {baseline_ckpt}; "
        f"candidate copied to {failed_archive}",
        flush=True,
    )
    return False
