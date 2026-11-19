"""Rescore explorer QA flags: baseline vs FP fine-tuned verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from treelidarnet.explorer.qa_flags import default_flags_path, load_flags
from treelidarnet.models.crop_verifier import load_crop_verifier
from treelidarnet.paths import runs_dir
from treelidarnet.verifier import score_detection


@dataclass
class FlagEvalSummary:
    n_flags: int
    n_scored: int
    n_skip_crop: int
    threshold: float
    v2_mean_delta: float
    v2_improved: int
    v2_still_high: int
    v3_mean_delta: float
    v3_improved: int
    v3_still_high: int
    v3_export_still_high: int


def _default_ckpt(name: str) -> Path:
    return runs_dir() / name / "best.pt"


def eval_explorer_flags(
    *,
    flags_path: Path | None = None,
    v2_baseline: Path | None = None,
    v2_fp: Path | None = None,
    v3_baseline: Path | None = None,
    v3_fp: Path | None = None,
    threshold: float = 0.5,
    out_parquet: Path | None = None,
) -> tuple[pd.DataFrame, FlagEvalSummary]:
    flags_path = flags_path or default_flags_path()
    v2_baseline = v2_baseline or _default_ckpt("gent_crop_verifier_v2")
    v2_fp = v2_fp or _default_ckpt("gent_crop_verifier_v2_fp")
    v3_baseline = v3_baseline or _default_ckpt("gent_crop_verifier_v3_rgb")
    v3_fp = v3_fp or _default_ckpt("gent_crop_verifier_v3_rgb_fp")

    for label, p in (
        ("flags", flags_path),
        ("v2_baseline", v2_baseline),
        ("v2_fp", v2_fp),
        ("v3_baseline", v3_baseline),
        ("v3_fp", v3_fp),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    flags = load_flags(flags_path, active_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_v2_o = load_crop_verifier(v2_baseline, device)
    m_v2_n = load_crop_verifier(v2_fp, device)
    m_v3_o = load_crop_verifier(v3_baseline, device)
    m_v3_n = load_crop_verifier(v3_fp, device)

    rows: list[dict] = []
    n_skip = 0
    for r in flags.to_dict("records"):
        x, y = float(r["x"]), float(r["y"])
        seed = hash(str(r["flag_id"])) % (2**31)
        s_v2_o = score_detection(m_v2_o, x=x, y=y, device=device, seed=seed)
        s_v2_n = score_detection(m_v2_n, x=x, y=y, device=device, seed=seed)
        s_v3_o = score_detection(m_v3_o, x=x, y=y, device=device, seed=seed)
        s_v3_n = score_detection(m_v3_n, x=x, y=y, device=device, seed=seed)
        if s_v2_o == 0.0 and s_v2_n == 0.0 and s_v3_o == 0.0 and s_v3_n == 0.0:
            n_skip += 1
        stored_v2 = float(r["v2"]) if r.get("v2") is not None and pd.notna(r["v2"]) else None
        stored_v3 = float(r["v3"]) if r.get("v3") is not None and pd.notna(r["v3"]) else None
        rows.append(
            {
                "flag_id": r["flag_id"],
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "bev": float(r["bev"]) if r.get("bev") is not None else None,
                "export_v2": stored_v2,
                "export_v3": stored_v3,
                "baseline_v2": s_v2_o,
                "fp_v2": s_v2_n,
                "delta_v2": s_v2_o - s_v2_n,
                "baseline_v3": s_v3_o,
                "fp_v3": s_v3_n,
                "delta_v3": s_v3_o - s_v3_n,
            }
        )

    df = pd.DataFrame(rows)
    scored = df[(df["baseline_v2"] > 0) | (df["fp_v2"] > 0) | (df["baseline_v3"] > 0) | (df["fp_v3"] > 0)]
    n_scored = len(scored)

    v2_imp = int((scored["delta_v2"] > 0.01).sum()) if n_scored else 0
    v3_imp = int((scored["delta_v3"] > 0.01).sum()) if n_scored else 0
    v2_hi = int((scored["fp_v2"] >= threshold).sum()) if n_scored else 0
    v3_hi = int((scored["fp_v3"] >= threshold).sum()) if n_scored else 0
    v3_exp_hi = int((scored["export_v3"].fillna(1.0) >= threshold).sum()) if n_scored else 0

    summary = FlagEvalSummary(
        n_flags=len(flags),
        n_scored=n_scored,
        n_skip_crop=n_skip,
        threshold=threshold,
        v2_mean_delta=float(scored["delta_v2"].mean()) if n_scored else 0.0,
        v2_improved=v2_imp,
        v3_mean_delta=float(scored["delta_v3"].mean()) if n_scored else 0.0,
        v3_improved=v3_imp,
        v2_still_high=v2_hi,
        v3_still_high=v3_hi,
        v3_export_still_high=v3_exp_hi,
    )

    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)

    return df, summary


def print_flag_eval_report(summary: FlagEvalSummary, df: pd.DataFrame) -> None:
    print(f"Explorer FP flags: {summary.n_flags} active, {summary.n_scored} scored", flush=True)
    if summary.n_skip_crop:
        print(f"  (all-zero scores on {summary.n_skip_crop} rows — likely missing crop)", flush=True)
    print(
        f"v2: mean drop (baseline→fp) {summary.v2_mean_delta:+.3f} | "
        f"improved (Δ>0.01): {summary.v2_improved}/{summary.n_scored} | "
        f"still ≥{summary.threshold}: {summary.v2_still_high}",
        flush=True,
    )
    print(
        f"v3: mean drop {summary.v3_mean_delta:+.3f} | "
        f"improved: {summary.v3_improved}/{summary.n_scored} | "
        f"still ≥{summary.threshold}: {summary.v3_still_high} "
        f"(export v3 high at flag time: {summary.v3_export_still_high})",
        flush=True,
    )
    if len(df):
        worst = df.nlargest(10, "fp_v3")[["flag_id", "lon", "lat", "export_v3", "baseline_v3", "fp_v3", "delta_v3"]]
        print("\nTop 10 still-high v3 after fp fine-tune:", flush=True)
        print(worst.to_string(index=False), flush=True)