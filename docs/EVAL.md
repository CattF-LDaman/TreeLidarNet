# TreeLidarNet — Evaluation and post-processing

Defines how model outputs become **detections**, how overlapping chips merge, and how **metrics** are computed. Implementation target: `treelidarnet/metrics.py` + `treelidarnet/eval.py`.

---

## 1. Scope

| Stage | Input | Output |
|-------|--------|--------|
| Per-chip infer | Chip points or BEV tensor | Heatmap + optional offsets |
| Peak extract | Heatmap | Candidate `(x, y, score)` in EPSG:31370 |
| City merge | All chip candidates | Deduped detection list |
| Match | Deduped preds + eligible GT | TP / FP / FN / discoveries |
| Report | Match sets | JSON + stratified tables |

Primary success metrics (report **both** — see [LABELS.md](LABELS.md)):

- **Strict F1 @ 3 m** — benchmark against the register (unlisted trees hurt precision).
- **Audit F1 @ 3 m** — discoveries (>5 m from any GT) excluded from FP.

Eval JSON uses `tree_level_strict` and `tree_level_audit` (`treelidarnet/eval.py`).

---

## 2. Per-chip peak extraction

After forward pass, heatmap **H** has shape matching the 0.25 m BEV grid (200×200 for a 50 m chip).

| Parameter | Default | Notes |
|-----------|---------|-------|
| Activation | `sigmoid(H)` | Already applied in head |
| Local max filter | **2.0 m** radius (8 px at 0.25 m) | scipy `maximum_filter` or max-pool |
| Score threshold | **0.35** initial; tune on **val** for max F1 | Store chosen threshold in checkpoint meta |
| Top-K cap | **64** per chip | Prevent heatmap speckle explosions |
| Sub-pixel refine | If offset head present: `xy_cell + offset` | Optional v1 |

Peaks inherit chip origin to produce Lambert coordinates.

---

## 3. Score calibration (optional v1)

| Method | Use |
|--------|-----|
| Fixed threshold | Default |
| Val-tuned threshold | Single global τ saved in `best.pt` meta |
| Temperature scaling | v1.1 if scores are miscalibrated |

Do not tune threshold on **test**.

---

## 4. Non-maximum suppression (NMS)

### 4.1 Within-chip

After local maxima, apply **greedy NMS** on candidates:

| Parameter | Default |
|-----------|---------|
| NMS radius | **2.0 m** planimetric |
| Tie-break | Higher score wins |

### 4.2 Across overlapping chips (city merge)

Inference uses the same chip grid as training (50 m, stride 25 m) over the eval region.

1. Concatenate all chip-level detections with scores.
2. Sort globally by score descending.
3. Greedy NMS with radius **2.5 m** (slightly wider than within-chip to collapse duplicate peaks from overlap).
4. Output **deduped city detection list** with optional `source_chip_ids[]` for debug.

**Metric level:** Official **precision/recall/F1** are computed on this **deduped tree-level** list, not per-chip averages (per-chip metrics may be logged for debugging only).

---

## 5. Ground-truth set for eval

From `gent_trees.parquet`:

- `eligible == true`
- Tree centroid in eval region (test kaartbladen)
- Transform to EPSG:31370 for matching

Post-2015 plantings and excluded `beheerfase` are **not** in **G**.

---

## 6. Matching predictions to inventory

Let **P** = deduped predictions above threshold, **G** = eligible GT in region.

| Parameter | Default |
|-----------|---------|
| Match distance | **3.0 m** (primary); also report **2 m** and **5 m** |
| Algorithm | **Greedy** by descending score, or Hungarian (same results at typical densities) |
| One-to-one | Each GT matches **at most one** pred; each pred **at most one** GT |
| Unmatched preds | **FP** or **discovery** (see below) |
| Unmatched GT | **FN** |

Report **mean/median** planimetric error for matched pairs.

### Allee / cluster ambiguity

Each **`gis_id`** is a separate GT point. Matching is **per `gis_id`**, not per canopy cluster. Multiple preds within 3 m of the same GT: best score matches; others are FP (unless NMS already removed them).

---

## 7. Discoveries bucket

Unmatched predictions with **score ≥ discovery_threshold** (default same as τ) and **no** GT within **5 m** are labeled **`discovery`**, not counted as FP in the primary **audit** table.

| Report | Contents |
|--------|----------|
| **Strict** | Unmatched preds = FP (for benchmark comparison) |
| **Audit** | Unmatched preds within 5 m of any ineligible/post-2015 tree → `ambiguous`; beyond → `discovery` |

Export discoveries to `exports/discoveries_{run_id}.geojson` for manual QA (fields: `score`, `lon`, `lat`, `nearest_gis_id`, `nearest_dist_m`).

---

## 8. Metrics JSON schema

`eval.py` writes:

```json
{
  "version": 1,
  "run_id": "gent_bev_v1_20261110",
  "split": "test",
  "match_distance_m": 3.0,
  "score_threshold": 0.42,
  "gent_trees_parquet_sha256": "…",
  "checkpoint_sha256": "…",
  "task": "register_tree_detection",
  "label_scope": "stad_gent_managed_inventory_partial",
  "tree_level_strict": {
    "precision": 0.81,
    "recall": 0.76,
    "f1": 0.78,
    "tp": 1200,
    "fp": 280,
    "fn": 380,
    "matched_mean_dist_m": 1.4
  },
  "tree_level_audit": {
    "precision": 0.88,
    "recall": 0.76,
    "f1": 0.82,
    "tp": 1200,
    "fp": 235,
    "fn": 380,
    "matched_mean_dist_m": 1.4
  },
  "match_distance_sweep_m": [2, 3, 5],
  "discoveries": { "count": 45, "threshold_m": 5.0 },
  "stratified": {
    "has_stamomtrek": { "f1": 0.80, "n": 9000 },
    "no_stamomtrek": { "f1": 0.71, "n": 2100 }
  }
}
```

---

## 9. CHM baseline evaluation

The frozen CHM pipeline (SPEC appendix A) produces peaks in the same Lambert frame. Apply **the same** §3–6 merge and matching so neural and CHM numbers are comparable.

---

## 10. Stratified reports

Minimum stratifications for v1:

- `stamomtrek` present vs absent  
- Top 10 `sortiment` keys by GT count (merge tail into `other`)  
- Urban vs park heuristic (optional: OSM land use or road buffer distance)

---

## 11. CI regression gate (recommended)

| Check | Floor (initial) |
|-------|-----------------|
| CHM baseline F1 @ 3 m on golden fixture | Documented constant in `tests/fixtures/golden_metrics.json` |
| Matcher unit tests | Synthetic P/G with known TP/FP/FN |
| CRS round-trip | \|Δxy\| < 0.05 m after WGS84 ↔ 31370 |

---

## 12. Related documents

| Doc | Contents |
|-----|----------|
| [DATA.md](DATA.md) | Splits, eligibility, versioning |
| [SPEC.md](SPEC.md) | Model, preprocessing, CHM appendix |
| [VISUALIZATION.md](VISUALIZATION.md) | Map payload for matched preds |
