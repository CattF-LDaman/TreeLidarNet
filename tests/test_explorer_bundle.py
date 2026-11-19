import json

import numpy as np
import pandas as pd
import pytest

from treelidarnet.explorer.bundle import (
    BUNDLE_MAGIC,
    BUNDLE_VERSION,
    BuildBundleConfig,
    build_explorer_bundle,
)


def test_build_bundle_small(tmp_path):
    preds = {
        "predictions": [
            {
                "position": [3.725, 51.054],
                "score": 0.9,
                "verifier_scores": {"v2": 0.6, "v3": 0.7},
            },
            {
                "position": [3.726, 51.055],
                "score": 0.88,
                "verifier_scores": {"v2": 0.4, "v3": 0.8},
            },
        ]
    }
    inp = tmp_path / "preds.json"
    inp.write_text(json.dumps(preds))
    trees = pd.DataFrame(
        [
            {
                "gis_id": "t1",
                "eligible": True,
                "lon": 3.725,
                "lat": 51.054,
                "x": 103500.0,
                "y": 195800.0,
                "stamomtrek": 40.0,
            }
        ]
    )
    trees_path = tmp_path / "trees.parquet"
    trees.to_parquet(trees_path, index=False)
    out = tmp_path / "explorer"
    meta = build_explorer_bundle(
        BuildBundleConfig(
            input_path=inp,
            output_dir=out,
            trees_parquet=trees_path,
            mark_rgb=False,
        )
    )
    assert meta["detection_count"] == 2
    assert meta["register_count"] == 1
    det = (out / "detections.bin").read_bytes()
    assert det[:4] == BUNDLE_MAGIC
    assert int.from_bytes(det[4:8], "little") == BUNDLE_VERSION
    n = int.from_bytes(det[8:12], "little")
    assert n == 2
    cols = np.frombuffer(det[12:], dtype=np.dtype("<f4")).reshape(n, 7)
    assert cols[0, 2] == pytest.approx(0.9)
    assert cols[0, 5] >= 0.0