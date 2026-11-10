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
  "model_id": "gent-v1-pointnet",
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
