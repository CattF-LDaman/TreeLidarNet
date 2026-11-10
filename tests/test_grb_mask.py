import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from treelidarnet.data.chips import rasterize_heatmap
from treelidarnet.data.grb import GrbIndex, make_hard_negative_mask


def test_hard_negative_from_building(tmp_path: Path):
    x0, y0 = 200_000.0, 193_000.0
    poly = Polygon(
        [
            (x0 + 10, y0 + 10),
            (x0 + 30, y0 + 10),
            (x0 + 30, y0 + 30),
            (x0 + 10, y0 + 30),
        ]
    )
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(poly)}]}
    grb_dir = tmp_path / "grb"
    grb_dir.mkdir()
    (grb_dir / "gent_grb_buildings.geojson").write_text(json.dumps(fc), encoding="utf-8")
    (grb_dir / "gent_grb_roads.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    grb = GrbIndex.from_dir(grb_dir)
    heat = rasterize_heatmap([(x0 + 5, y0 + 5)], origin_x=x0, origin_y=y0)
    mask = make_hard_negative_mask(grb, origin_x=x0, origin_y=y0, heatmap=heat)
    assert mask.max() == 1.0
    assert mask.mean() > 0.05
    peak_y, peak_x = np.unravel_index(np.argmax(heat), heat.shape)
    assert mask[peak_y, peak_x] == 0.0
