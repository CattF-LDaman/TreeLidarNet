"""Gate detections to chips that have valid RGB (v3 verifier QA)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from treelidarnet.data.chips import chip_id
from treelidarnet.data.crops import chip_origin_for_point
from treelidarnet.paths import chips_dir
from treelidarnet.postprocess import Detection

_DEBUG_LOG = Path(__file__).resolve().parents[2] / ".cursor" / "debug-f13d06.log"
_SESSION = "f13d06"


def _agent_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": _SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": data.get("runId", "filter"),
    }
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # endregion


def chip_has_valid_rgb(chip_id_value: str) -> bool:
    path = chips_dir() / f"{chip_id_value}.npz"
    if not path.is_file():
        return False
    try:
        with np.load(path, mmap_mode="r") as data:
            return (
                "rgb" in data.files
                and "xyz" in data.files
                and len(data["rgb"]) == len(data["xyz"])
                and len(data["xyz"]) > 0
            )
    except OSError:
        return False


def filter_detections_on_rgb_chips(
    detections: list[Detection],
    *,
    run_id: str = "filter",
) -> tuple[list[Detection], dict[str, int]]:
    """Keep detections whose 50 m tile has a valid rgb array."""
    cache: dict[str, bool] = {}
    kept: list[Detection] = []
    no_chip = 0
    no_rgb = 0
    for d in detections:
        x0, y0 = chip_origin_for_point(d.x, d.y)
        cid = chip_id(x0, y0)
        if cid not in cache:
            cache[cid] = chip_has_valid_rgb(cid)
        if cache[cid]:
            kept.append(d)
        elif not (chips_dir() / f"{cid}.npz").is_file():
            no_chip += 1
        else:
            no_rgb += 1
    stats = {
        "input": len(detections),
        "rgb_kept": len(kept),
        "dropped_no_rgb": no_rgb,
        "dropped_no_chip": no_chip,
        "unique_chips_checked": len(cache),
        "unique_chips_with_rgb": sum(1 for v in cache.values() if v),
    }
    _agent_log(
        hypothesis_id="A",
        location="rgb_chip_gate.py:filter_detections_on_rgb_chips",
        message="rgb gate stats",
        data={**stats, "runId": run_id},
    )
    return kept, stats