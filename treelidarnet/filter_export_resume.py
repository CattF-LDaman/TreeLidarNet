"""Resume partial live verifier scoring from status + verified_predictions.json."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import orjson

from treelidarnet.postprocess import Detection


def load_scoring_resume(
    *,
    status_path: Path,
    live_pred_path: Path,
    preds: list[Detection],
    compare: bool,
) -> tuple[int, list[Detection], list[dict[str, float]] | None]:
    """
    Return (resume_done, scored_prefix, dual_prefix).
    When resume_done == len(preds) and phase was scoring, caller may skip forward only.
  """
    if not status_path.is_file():
        return 0, [], [] if compare else None
    try:
        status = orjson.loads(status_path.read_bytes())
    except orjson.JSONDecodeError:
        return 0, [], [] if compare else None
    phase = status.get("phase")
    done = int(status.get("scored") or status.get("done") or 0)
    if done <= 0:
        return 0, [], [] if compare else None
    if phase == "done":
        done = max(done, len(preds))
    if not live_pred_path.is_file():
        return 0, [], [] if compare else None
    try:
        payload = orjson.loads(live_pred_path.read_bytes())
    except orjson.JSONDecodeError:
        return 0, [], [] if compare else None
    rows = payload.get("predictions") or []
    n = min(done, len(rows), len(preds))
    if n <= 0:
        return 0, [], [] if compare else None
    scored: list[Detection] = []
    dual: list[dict[str, float]] = []
    is_compare = bool(payload.get("compare") and compare)
    for i in range(n):
        row = rows[i]
        p = preds[i]
        vs = row.get("verifier_scores")
        if is_compare and isinstance(vs, dict):
            v2 = float(vs.get("v2", 0.0))
            v3 = float(vs.get("v3", 0.0))
            dual.append({"v2": v2, "v3": v3})
            scored.append(replace(p, verifier_score=v3))
        else:
            v = row.get("verifier_score")
            prob = float(v) if v is not None else 0.0
            scored.append(replace(p, verifier_score=prob))
    dual_out: list[dict[str, float]] | None = dual if is_compare else None
    print(f"  resume: {n}/{len(preds)} scores loaded from {live_pred_path.name}", flush=True)
    return n, scored, dual_out