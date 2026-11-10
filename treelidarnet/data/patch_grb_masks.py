"""Update hard_negative in existing chip npz files (no LAZ rebuild)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.grb import GrbIndex, make_hard_negative_mask
from treelidarnet.paths import chips_dir, manifests_dir


def patch_chips_grb(
    *,
    grb_dir: Path,
    index_parquet: Path | None = None,
    road_buffer_m: float = 2.0,
    limit: int | None = None,
) -> None:
    index_parquet = index_parquet or (manifests_dir() / "chips_index.parquet")
    df = pd.read_parquet(index_parquet)
    grb = GrbIndex.from_dir(grb_dir, road_buffer_m=road_buffer_m)
    rows = df.to_dict("records") if limit is None else df.head(limit).to_dict("records")
    for i, row in enumerate(rows, 1):
        cid = str(row.get("chip_id", ""))
        if not cid:
            cid = f"gent_50m_{int(row['origin_x'])}_{int(row['origin_y'])}"
        path = chips_dir() / f"{cid}.npz"
        if not path.is_file():
            continue
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        heat = arrays["heatmap"].astype(np.float32)
        x0 = float(arrays["origin"][0]) if "origin" in arrays else float(row["origin_x"])
        y0 = float(arrays["origin"][1]) if "origin" in arrays else float(row["origin_y"])
        hn = make_hard_negative_mask(grb, origin_x=x0, origin_y=y0, heatmap=heat)
        arrays["hard_negative"] = hn
        np.savez_compressed(path, **arrays)
        if i % 500 == 0:
            mean_hn = float(hn.mean())
            print(f"patched {i}/{len(rows)} mean_hard_neg={mean_hn:.3f}", flush=True)
    print(f"Done patching {len(rows)} chips", flush=True)
