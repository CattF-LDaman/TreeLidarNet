# TreeLidarNet — Inference map (Deck.GL)

Visual target: the same **dark MapLibre + Deck.GL overlay** language as Luwe Daman’s [Trees in Ghent](https://luwe.be/playground/trees-in-ghent) playground and HugATree’s production map — extended for **model outputs**, not just open data scatter.

---

## Stack

| Layer | Choice |
|-------|--------|
| Basemap | MapLibre GL JS + CARTO `dark-matter` style |
| Overlay | `deck.MapboxOverlay` with `interleaved: true` (see HugATree heatmap note below) |
| Deck.GL | Pin major version across apps (HugATree uses 9.2.x; align TreeLidarNet with that) |
| Data API | JSON (gzip) inference bundles; optional COG/mvt later |

Reference implementations:

- [`~/Dev/LuweDaman/portfolio/templates/playground/trees_in_ghent.html`](../../Dev/LuweDaman/portfolio/templates/playground/trees_in_ghent.html) — minimal scatter + heatmap  
- [`~/Dev/HugATree/static/js/map.js`](../../Dev/HugATree/static/js/map.js) — filters, tooltips, secondary highlight layers, hug heatmap  

---

## Map layers (inference mode)

Render order (bottom → top):

| Layer ID | Deck type | Source | Purpose |
|----------|-----------|--------|---------|
| `lidar-ground` | optional `PointCloudLayer` or precomputed `Tile3DLayer` | Subsampled LAZ chip (web-friendly) | Context at high zoom — **cap point count** for performance |
| `gt-trees` | `ScatterplotLayer` | Gent inventory subset in view | Green markers; ground truth |
| `pred-trees` | `ScatterplotLayer` | Model detections | Confidence → color; matched/unmatched styling |
| `pred-heatmap` | `HeatmapLayer` (toggle) | Detections or class probability grid | Density QA |
| `focus-ring` | `ScatterplotLayer` | Selected tree / chip | Same halo pattern as HugATree filter highlight |

### Ground truth styling (match HugATree)

```javascript
// Sized vs unknown trunk — portfolio emerald palette
getFillColor: (d) =>
  d.hasSize ? [52, 211, 153, 175] : [101, 163, 122, 170];
radiusUnits: "meters";
getRadius: (d) => d.radiusMeters; // from stamomtrek heuristic in tree_map_cache
```

### Prediction styling

| State | Fill | Stroke | Radius |
|-------|------|--------|--------|
| High confidence | `[56, 189, 248, 200]` | `[125, 211, 252, 180]` | `score * baseRadius` or fixed 3 m |
| Matched to `gis_id` | same + check badge in tooltip | link to nearest GT | — |
| False positive | `[251, 146, 60, 190]` | warm ring | 2.5 m |
| False negative (GT with no pred) | pulse ring `[244, 63, 94, …]` | on `gt-trees` overlay | 1.65× radius (HugATree filter pattern) |

### Interleaved mode

When the heatmap is on, HugATree sets `interleaved: false` on the overlay so blending behaves. Mirror that toggle logic in TreeLidarNet.

---

## Payload shape (inference JSON)

Compatible with HugATree scatter records plus model fields:

```json
{
  "version": 1,
  "model_id": "gent-crop-verifier-v1",
  "inference_at": "2026-11-10T00:00:00Z",
  "chip_id": "gent_50m_3k12_8j44",
  "gt_trees": [
    {
      "position": [3.725, 51.054],
      "gis_id": "fysgro/tep335529",
      "slug": "fysgro-tep335529",
      "radiusMeters": 4.2,
      "hasSize": true,
      "sortiment": "Acer pseudoplatanus",
      "matched_pred_index": 0
    }
  ],
  "predictions": [
    {
      "position": [3.7251, 51.0541],
      "score": 0.92,
      "matched_gis_id": "fysgro/tep335529",
      "radiusMeters": 3.0
    }
  ],
  "metrics": {
    "precision_at_3m": 0.81,
    "recall_at_3m": 0.76,
    "f1_at_3m": 0.78
  }
}
```

Serve gzip (`application/json` + `Content-Encoding: gzip`) like HugATree `trees_map.json.gz`.

For viewport-based loading, add a lightweight **`GET /inference/tiles?bbox=`** that returns only predictions intersecting the map bounds.

### City-scale LOD and loading contract

**Gent demo (`viewer/ghent.html`, `ghent_live.html`):** full GT + prediction scatter in one bundle (HugATree / Luwe playground pattern) — amber rings are the **product**, green register is optional.

For other municipalities or much larger runs, full-city scatter may be too heavy:

| Zoom | Behavior |
|------|----------|
| **z &lt; 13** | `HeatmapLayer` or server-side **H3/ grid counts** only (no per-tree scatter) |
| **13–15** | Viewport `bbox` API returns max **4,000** points (preds + GT); priority: FN/FP/matched first |
| **≥ 15** | Full detail in bbox; chip mode loads single `chip_id` JSON for QA |

**`GET /inference/tiles` query params (v1 contract):**

| Param | Required | Description |
|-------|----------|-------------|
| `bbox` | yes | `west,south,east,north` WGS84 |
| `layers` | no | `gt`, `pred`, `fp`, `fn` (default all) |
| `run_id` | no | Checkpoint run id; default latest |

Response: gzip JSON `{ "gt_trees": [...], "predictions": [...], "truncated": true, "count_cap": 4000 }`.

Static viewer v1 may ship **chip-only** mode; city bbox mode is M3+.

---

## UI chrome

Reuse Luwe **map demo panel** patterns:

- Glass panel (`rgba(13,17,23,0.72)`, blur, emerald section labels).
- Status line: loading LAZ chip → running inference → N predictions.
- Toggles: heatmap, show false positives only, show unmatched GT.
- Legend: GT green, prediction cyan, FP orange, FN rose ring.
- Footer credits: Digitaal Vlaanderen, Stad Gent, OSM, CARTO.

Optional integration paths:

- **Standalone** static viewer in `TreeLidarNet/viewer/` (fast iteration on models).
- **Luwe Daman playground page** “LiDAR trees (Gent)” once API is stable.
- **HugATree** admin-only overlay comparing municipal DB to model gaps (future).

---

## End-product explorer (Flask, full dataset)

**Page:** **http://127.0.0.1:8765/explorer**  
**Server:** `make explorer-serve` (Flask — crop API + map; preferred after full dual verify)

| Piece | Stack |
|-------|--------|
| Data | `make explorer-bundle` → `exports/inference/explorer/` (`detections.bin`, `register.bin`, `meta.json`) |
| Map | MapLibre + deck.gl; columnar load (no 25k cap); combine presets (v2/v3 OR/AND, register split, discoveries) |
| Click | Same `/api/crop/wgs84` + Three.js panel as crop QA |

Legacy Starlette crop QA (`make viewer-crop-qa`, [`viewer/crops.html`](../viewer/crops.html)) remains for live polling workflows.

## Crop QA viewer (register + BEV + verifier → 3D)

**Page:** [`viewer/crops.html`](../viewer/crops.html)  
**Server:** `make viewer-crop-qa` (loads `runs/gent_crop_verifier_v2/best.pt` for click scoring) — **http://127.0.0.1:8765**

| Piece | Stack |
|-------|--------|
| Map | MapLibre + deck.gl — ring **width/color** = BEV score; **green/red fill halo** = verifier when present |
| Register | `GET /api/register/all` (full city, HugATree-style) |
| Predictions | `GET /api/predictions?path=` or **live** poll `GET /api/live/verified/predictions` |
| Live verify | `filter-export --live --bev-min 0.9` writes `exports/inference/live/verified_*.json`; enable **Live verifier scoring** on the map |
| Crop on click | `GET /api/crop/wgs84?lon&lat` — extract + optional **verifier_score** in API response |
| 3D panel | Three.js point cloud + crop **radius ring**; score bars for BEV / verifier |

**Live workflow**

```bash
# Terminal 1
make viewer-crop-qa

# Terminal 2 (subset — use --limit while experimenting)
make filter-export-live-center
# or full high-conf slice:
# uv run treelidarnet filter-export --live --bev-min 0.9 \
#   --input exports/inference/gent_city_gent_canopy_v2.json.gz \
#   --output exports/inference/gent_city_gent_canopy_v2_verified.json.gz \
#   --verifier runs/gent_crop_verifier_v2/best.pt
```

On the map: enable **Live verifier scoring**, set **Min BEV** (e.g. 0.90), tune **Min verifier** to hide low scores without re-running export.

---

## LinkedIn verifier showcase reel (MP4)

Offline reel: map zoom on a detection, then a long **downward isometric orbit** — alternating soft **RGB splats** and **height-colored points** (same palette as the crop 3D viewer). Requires **ffmpeg** and scored rows in `verified_predictions.json`. Defaults: `--cloud-frames 120`, `--cloud-rotations 1.4`.

```bash
uv run python scripts/render_verifier_linkedin_reel.py \
  --verified exports/inference/live/verified_predictions.json \
  --n 20 --bev-min 0.85 --verifier-min 0.85 --min-spacing-m 400 \
  --output exports/inference/linkedin_verifier_reel.mp4
```

Writes `linkedin_verifier_reel_manifest.json` with the 20 picks (scores, chip_id, frame ranges). Only detections on chips that already have `rgb` in the npz are eligible.

---

## Point cloud preview

Full 16 pts/m² LAZ is too heavy for the browser. Options (pick one per chip view):

1. **Server subsample** — random or voxel-grid downsample to ≤200k points; binary PLY or LAS in CDN.
2. **deck.gl `PointCloudLayer`** — `getColor` from classification or height above ground.
3. **Skip live points** — show DSM hillshade WMS from OpenLiDAR as context only (lowest effort).

Default for v1: **no live point cloud**; scatter-only inference map. Add preview in v2 when chip export pipeline exists.

---

## Accessibility and performance

- Keyboard-focusable layer toggle controls; tooltips use `role="status"`.
- Target 60 fps at city zoom 12–16 on a mid-range laptop — keep scatter <5k points in view without clustering.
- `prefers-reduced-motion`: disable FN pulse animation.
