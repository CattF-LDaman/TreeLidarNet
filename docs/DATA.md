# TreeLidarNet — Data sources and alignment

## Overview

TreeLidarNet joins two open datasets:

1. **Supervision** — municipal tree *locations* (and attributes) from Gent.
2. **Features** — DHMV II raw LiDAR point clouds from Digitaal Vlaanderen.

The hard part is not downloading either source; it is **aligning semantics** (inventory point ≠ crown centroid), **choosing spatial units** (500 m LAZ tiles vs ~68k labels), and **holding out geography** so metrics mean something on unseen streets.

---

## Ground truth — Stad Gent *Locaties bomen Gent*

| Item | Detail |
|------|--------|
| Portal | [data.stad.gent](https://data.stad.gent/) |
| Dataset | [locaties-bomen-gent](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) |
| GeoJSON export | `https://data.stad.gent/api/explore/v2.1/catalog/datasets/locaties-bomen-gent/exports/geojson` |
| Records API | `https://data.stad.gent/api/explore/v2.1/catalog/datasets/locaties-bomen-gent/records` |
| Approx. count | ~68,240 trees (see HugATree import) |
| License | Stad Gent open data — **credit in UI and publications** |

### Canonical identifier

Use **`gis_id`** (example: `fysgro/tep335529`) as the primary key, URL slug, and join key to HugATree. Do **not** rely on the slim Luwe Daman playground export (`trees_ghent.json.gz`) alone; it omits `gis_id`.

Import logic and field usage are documented in [HugATree `docs/DATA.md`](../../Dev/HugATree/docs/DATA.md).

### Fields used for ML and evaluation

| Gent field | ML usage |
|------------|----------|
| `gis_id` | Stable tree ID, error analysis, map links to HugATree |
| Point geometry | WGS84 `lon`, `lat` — transform to Lambert 72 for LiDAR |
| `sortiment` | Species/cultivar label; optional auxiliary task or stratified metrics |
| `stamomtrek` | Trunk circumference (cm); weak prior on crown extent / height |
| `straatnaam` | Tile/street-level splits, qualitative review |
| `eigenaar`, `beheerfase`, `aanlegjaar`, `plantwijze` | Metadata; not required for v1 detection |

### What the labels are *not*

- **Not** a full city tree census (private trees, young whips, and park understory may be missing).
- **Not** crown polygons or stem diameters from LiDAR — the point is typically the **managed tree location**, which may sit near the trunk, not the geometric crown center.
- **Not** synchronized in time with LiDAR (inventory updates vs 2013–2015 flight campaign).

Evaluation must use **distance-based matching** (see [EVAL.md](EVAL.md)) rather than pixel-perfect masks unless we derive masks from points.

### Label eligibility (temporal and administrative)

DHMV II was flown **2013–2015**; Gent’s inventory is a **living** register. Only **eligible** trees become training/eval supervision; ineligible rows stay in `gent_trees.parquet` for audit.

| Rule | Default (v1) | Rationale |
|------|----------------|-----------|
| `aanlegjaar` parseable as year **> 2015** | **Exclude** from GT heatmaps and primary metrics (`eligible: false`) | No LiDAR signature at planting time |
| `aanlegjaar` empty or **≤ 2015** | **Eligible** | Assume present during campaign or unknown |
| `aanlegjaar` invalid / non-numeric | **Eligible** (flag `aanlegjaar_unknown: true`) | Gent data quality varies |
| `beheerfase` contains removal cues (e.g. *Afbraak*, *Verwijderd*, *Sloop* — exact Gent vocab TBD at import) | **Exclude** | Tree likely absent in 2013–2015 |
| `sortiment` = `Te Bepalen` | **Eligible** | Position still useful; stratify in metrics |
| Geometry not `Point` or missing `gis_id` | **Drop** at import | Same as HugATree |

**Sensitivity run (optional):** Report a second metric block that treats post-2015 plantings as FN-only (*stress test*), not as training positives. Do not mix that cohort into the default F1 without labeling the table.

Add columns on export: `eligible` (bool), `ineligible_reason` (nullable string).

### Clustered labels (allees and rows)

Multiple `gis_id` points can fall within **3 m** when `plantwijze` is *In rij* or similar. Training still places **one Gaussian per eligible `gis_id`**; overlapping Gaussians are allowed. Evaluation matching is defined in [EVAL.md](EVAL.md) (per-`gis_id` matching, not one-per-cluster).

---

## Features — DHMV II LiDAR (EODaS OpenLiDAR)

| Item | Detail |
|------|--------|
| Name | LiDAR Digitaal Hoogtemodel Vlaanderen **II** (DHMV II; sometimes verbally “DHV II”) |
| Campaign | 2013–2015, Flanders + Brussels |
| Portal | [EODaS OpenLiDAR](https://remotesensing.vlaanderen.be/apps/openlidar/) |
| Catalog | [datavindplaats — ruwe remote sensing data](https://www.vlaanderen.be/datavindplaats/catalogus/lidar-digitaal-hoogtemodel-vlaanderen-ii-ruwe-remote-sensing-data) |
| Format | **LAZ** (LAS 1.4 compressed), ~500 m × 500 m tiles |
| Density | ≥16 pts/m² effective (overlapping flight lines; tile-level min ~8 pts/m²) |
| Manual | [EODaS Open LiDAR handleiding (PDF)](https://remotesensing.vlaanderen.be/apps/openlidar/EODaS_openlidar_Handleiding.pdf) |

### Per-point attributes (DHMV II)

Typical LAZ records include:

| Attribute | Use in TreeLidarNet |
|-----------|---------------------|
| X, Y, Z | Geometry; **Z in TAW** (Belgian reference level) |
| Classification | Provider classes: ground (2), unclassified/non-ground (1), water (9) — start features / weak labels |
| Intensity | Surface reflectance proxy |
| Return number / number of returns | Vegetation structure |
| RGB | Co-registered ortho colors (optional input channel) |
| Point source ID | Flight strip — useful for fixing strip artifacts in QA |

### Horizontal and vertical reference

- **Planimetric:** Lambert 72 — **EPSG:31370** (standard for DHMV products and GRB).
- **Height:** **TAW** (official DHMV vertical datum).
- Gent inventory points arrive in **WGS84**; all training windows must use a single projected CRS (`pyproj` / `rasterio.warp`).

### Download workflow

1. Select **LiDAR DHMV II LAZ tiles** (individual tiles) or **kaartblad** (`.tar` of all tiles in an NGI sheet) in OpenLiDAR.
2. For automation: export CSV of tile URLs (portal supports bulk link export).
3. Store raw LAZ under `data/raw/laz/` mirroring tile naming from the portal — **do not commit** multi-terabyte archives to git.
4. For Ghent pilot: prefetch tiles overlapping the municipal boundary (roughly NGI sheets covering Gent; exact tile list is a build artifact in `data/manifests/gent_laz_tiles.csv`).
5. **Availability check:** Not every tile may be downloadable (EODaS may reflect [military air-photography constraints](https://remotesensing.vlaanderen.be/apps/openlidar/EODaS_openlidar_Handleiding.pdf)); record `download_ok` and `http_status` in `tile_index.parquet`. Gent centre is usually unrestricted — verify in manifest build, do not assume 100% coverage.

### Tile seams and point density

Flight lines overlap **~50%** → local density often exceeds 16 pts/m²; **500 m LAZ tiles** can duplicate points at boundaries when multiple tiles load for one chip.

| Policy | v1 default |
|--------|------------|
| Merge tiles for chip bbox | Union points; **dedupe** exact duplicate `(x,y,z,return)` within **0.01 m** tolerance |
| Density cap (optional) | Voxel-centroid subsample to **8 pts/m²** equivalent per chip for training |
| Per-chip stats | Store `point_count`, `mean_pts_per_m2`, `strip_count` in `chips_index` |
| Strip normalization | Keep `point_source_id` in features; optional per-strip cap at train time (see SPEC §5.6) |

### Derived rasters (optional baselines)

DHMV II DSM/DTM 1 m rasters (via [download.vlaanderen.be](https://download.vlaanderen.be/)) support **classical baselines** (CHM → local maxima) without touching raw LAZ. They are useful for sanity checks and ablation, not a replacement for the point-cloud model.

---

## Spatial indexing and manifests

Recommended manifest tables (Parquet or SQLite):

| Manifest | Purpose |
|----------|---------|
| `gent_trees.parquet` | Full Gent import with `gis_id`, WGS84, Lambert XY, attributes, `eligible` |
| `tile_index.parquet` | LAZ tile bounds (Lambert), file path, download URL, checksum, `download_ok` |
| `tree_tile_map` | Many-to-many: which trees fall in which LAZ tiles (buffered) |
| `chips_index` | Training/eval chip IDs, centroid, split, source tiles, QC stats |

**Buffer rule:** When assigning a tree to tiles, expand the tree point by **≥15 m** (half canopy + planimetric error) so edge trees load neighboring tiles.

### Dataset versioning (pinned exports)

Gent’s GeoJSON changes over time (HugATree may refresh daily). ML experiments must pin:

| Field | Location |
|-------|----------|
| `gent_export_url` | `split_meta.json` |
| `gent_export_fetched_at` | ISO 8601 UTC |
| `gent_trees_parquet_sha256` | Hash of `gent_trees.parquet` used to build chips |
| `gent_record_count` / `gent_eligible_count` | Counts after label eligibility |
| `lidar_epoch` | Constant `"DHMV_II_2013-2015"` for v1 |

Rebuilding chips after a new Gent pull creates a **new dataset generation**; do not compare F1 across generations without noting the hash change.

---

## Train / validation / test splits

Avoid random point splits — they leak spatial structure. **Street hash alone is insufficient** because chips overlap and share canopy context.

### Split unit (v1 default)

**Primary unit: NGI kaartblad** (map sheet) assigned wholly to `train`, `val`, or `test`.

| Rule | Detail |
|------|--------|
| Assignment | All LAZ tiles and chips whose centroid falls in a held-out sheet → that split only |
| `gis_id` invariant | No `gis_id` may appear in chips from more than one split |
| Streets | Used for **stratification / reporting**, not as the sole hold-out key |
| Val vs test | ~15% sheets → val, ~15% → test, remainder train (exact list in `split_meta.json`) |

**Chip inheritance:** A chip’s split is the split of its **kaartblad** (not majority vote of trees inside the chip).

**Leakage controls:**

- Overlapping 50 m chips with 25 m stride are allowed **within** the same split only.
- Do not use test-sheet tiles when computing global normalization statistics (use train only).

### `split_meta.json` schema (v1)

```json
{
  "version": 1,
  "split_unit": "ngi_kaartblad",
  "lidar_epoch": "DHMV_II_2013-2015",
  "gent_export_url": "https://data.stad.gent/api/explore/v2.1/catalog/datasets/locaties-bomen-gent/exports/geojson",
  "gent_export_fetched_at": "2026-11-10T12:00:00Z",
  "gent_trees_parquet_sha256": "…",
  "gent_record_count": 68240,
  "gent_eligible_count": 65100,
  "label_eligibility": {
    "aanlegjaar_max_lidar_year": 2015,
    "exclude_beheerfase_patterns": ["Afbraak", "Verwijderd"]
  },
  "sheets": {
    "train": ["31", "32", "…"],
    "val": ["29"],
    "test": ["30"]
  },
  "seed": 42
}
```

### Alternate strategies (document if used)

| Strategy | When |
|----------|------|
| North/south bisect | Few sheets only; document bisect line in Lambert |
| `gis_id`-disjoint manual list | Small pilot subsets before full sheet manifest |

---

## HugATree reuse

| HugATree asset | TreeLidarNet use |
|----------------|------------------|
| `hugatree/services/tree_import.py` | Same Gent GeoJSON import |
| `trees_map.json` scatter shape | Deck.GL ground-truth layer (`position`, `radiusMeters`, `gis_id` slug) |
| `species_lookup_key` / vernacular | Tooltips and species-stratified metrics |

---

## Attribution (required)

**UI and papers must credit:**

- [Digitaal Vlaanderen / EODaS](https://remotesensing.vlaanderen.be/apps/openlidar/) — DHMV II LiDAR  
- [Stad Gent Open Data](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) — tree inventory  
- [OpenStreetMap](https://www.openstreetmap.org/copyright) and [CARTO](https://carto.com/attributions) — basemap (visualization)

Redistribution of derived artifacts: see [LEGAL.md](LEGAL.md).
