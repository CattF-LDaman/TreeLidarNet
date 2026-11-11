# TreeLidarNet — Product & research specification

Version 0.2 · Human-readable · Ghent pilot · PyTorch

---

## 1. One-liner

**TreeLidarNet** learns to **find trees in DHMV II airborne LiDAR** using **Stad Gent’s municipal tree inventory** as weak-but-valuable supervision, trains in **PyTorch**, and validates both with **geographic metrics** and a **Deck.GL map** in the spirit of HugATree and Luwe Daman’s Ghent playground.

---

## 2. Problem statement

### 2.1 What we are solving

Given a patch of DHMV II point cloud (or a feature tensor derived from it), predict **where tree-like canopy appears in LiDAR** in planimetric space — **instance locations** (xy) with **confidence** — across **Ghent**, using the municipal register only as **soft positive** supervision (v2). Unlabeled canopy is **not** trained as negative ([TASK_GHENT_BROAD.md](TASK_GHENT_BROAD.md)).

Register **strict / audit** metrics remain for sanity checks ([LABELS.md](LABELS.md)).

### 2.2 Why Gent first

| Advantage | Detail |
|-----------|--------|
| Dense labels | ~68k georeferenced trees with stable `gis_id` |
| Existing tooling | HugATree import, map cache, Deck.GL UX |
| DHMV II coverage | Full LiDAR over Gent (2013–2015 epoch) |
| Error analysis | Link predictions to real tree profile pages (`/trees/<slug>/`) |

### 2.3 What success looks like

- **Primary (canopy v2):** Citywide inference with **credible discovery layer** (audit metrics + map QA); stable peaks in parks and streets without striping.
- **Secondary:** Held-out **audit F1 @ 3 m** vs register; **strict F1** documented but not the north star.
- **Qualitative:** Deck.GL map — predictions-first; register overlay optional (HugATree-style).
- **Engineering:** Reproducible train/eval scripts; one command to run inference on a named chip and export map JSON.

### 2.4 Explicit non-goals (v1)

- Species classification from LiDAR alone (Gent `sortiment` is available for *evaluation* stratification, not required as output).
- Flanders-wide training (Ghent pilot only; architecture must scale later).
- Real-time inference over the full city in the browser.
- Replacing HugATree’s inventory — this project **audits and augments** open data, it does not publish a new canonical census.
- Multi-temporal change detection (single LiDAR epoch).

---

## 3. Scientific framing

### 3.1 Label noise and semantic gap

Municipal points are **administrative tree locations**, not LiDAR-derived stem bases. Expect:

- Horizontal offset **0–5 m** (planting stake vs crown projection).
- Missing trees (private, small, recent plantings after 2015).
- **False negatives** in evaluation when the model finds a real tree not in the DB — track separately as *discoveries*, not pure FP.

### 3.2 Task formulations (phased)

| Phase | Formulation | Output | Role |
|-------|-------------|--------|------|
| **A — Baseline** | CHM from DSM−DTM, local maxima + height threshold | xy peaks | Sanity check, metric floor |
| **B — Semantic** | Point-wise classes: `{ground, vegetation, building, other}` | Per-point logits | Feature learning |
| **C — Detection** | Heatmap / offset regression of tree centers in 2.5D grid | xy + score | Primary v1 target |
| **D — Instance** | Balloon clustering or graph partition on point embeddings | Instance masks | v2+ |

**v1 deliverable:** Phase **A + C** with a shared preprocessing pipeline; Phase B optional if timeboxed.

### 3.3 Matching protocol (evaluation)

Full pipeline: [EVAL.md](EVAL.md) (peak extract → NMS → city merge → match).

| Parameter | Default |
|-----------|---------|
| Match distance | **3.0 m** planimetric (EPSG:31370); report **2 m** and **5 m** too |
| Multiple preds per GT | Allow only best-scoring match (others count as FP) |
| Score threshold | Tune on val for max F1; report PR curve |
| Metric level | **Tree-level** deduped preds (not per-chip mean) |

Report **precision, recall, F1**, and **mean distance** for matched pairs. Stratify by `stamomtrek` present/absent and by top-N species.

### 3.4 Geographic splits and leakage

**Split unit (v1):** whole **NGI kaartblad** sheets → train / val / test. No `gis_id` appears in more than one split. Chips inherit split from their sheet.

Street name is for **reporting only**, not sole hold-out. Overlapping 50 m chips (25 m stride) stay within one split.

Schema and versioning: [DATA.md](DATA.md) (`split_meta.json`).

---

### 3.5 Seasonality

DHMV II was acquired **winter, leaf-off** (2013–2015). Deciduous crowns look smaller in CHM; conifers and street trees dominate signatures. Do not expect summer-ortho crown diameters.

---

## 4. Data

Full detail: [DATA.md](DATA.md).

**Summary:**

- **LiDAR:** DHMV II LAZ via [EODaS OpenLiDAR](https://remotesensing.vlaanderen.be/apps/openlidar/).
- **Labels:** [locaties-bomen-gent](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) → `gis_id`, WGS84 point, attributes.
- **CRS:** Train in **EPSG:31370** + TAW heights; expose WGS84 for map layers only.

---

## 5. Preprocessing pipeline

### 5.1 Chip geometry

| Parameter | Default | Notes |
|-----------|---------|-------|
| Chip size | **50 m × 50 m** | Fits GPU memory; ~125k–400k points at 16 pts/m² |
| Stride (train) | 25 m | Overlap for augmentation |
| Origin | Aligned to Lambert meter grid | Reproducible chip IDs |

For each chip:

1. Load covering LAZ tile(s); crop to bbox; dedupe seam points ([DATA.md](DATA.md)).
2. Normalize Z → **height above ground** (`h_agl`) via the ground fallback chain (below).
3. Mask **LAS class 9** (water) pixels from loss where present (e.g. Scheldt corridors).
4. Optional: remove buildings using coarse GRB footprint mask (v2).

#### Ground height fallback chain

Apply in order until a ground value exists for each planimetric cell (or point):

| Step | Source | When |
|------|--------|------|
| 1 | **DHMV II DTM 1 m** (official raster via download.vlaanderen.be) | Default — sampled at xy |
| 2 | **LAS class 2** points | Per-cell median ground if DTM nodata |
| 3 | **Lowest return** in 2 m xy window | Urban fallback under sparse class 2 |
| 4 | Chip minimum Z | Last resort; flag `ground_unreliable: true` |

Per-chip QC in `chips_index`:

- `ground_class2_fraction` — share of points with class 2  
- `ground_unreliable` — bool if step 4 used for >10% of chip area  
- Training: **skip** or **down-weight** (`sample_weight=0.25`) unreliable chips  

CHM baseline uses the **same** DTM for fair comparison (Appendix A).

### 5.2 Training targets (detection head)

Rasterize tree locations onto a **0.25 m** grid inside the chip:

- **Center heatmap:** Gaussian at each eligible tree xy; **per-tree σ** from register `stamomtrek` (girth in cm): `σ = clamp(0.9 + 0.015 × st, 1.0, 2.5)` m; **1.0 m** if missing/invalid (`treelidarnet.data.chips.heatmap_sigma_m`).
- **Ignore mask:** All zeros. Every stake inside its 50 m tile is a training target. The old annulus 8–15 m from the edge is cleared: these tiles do not overlap, so a stake there was never a positive.

### 5.3 Augmentation (train)

- Random yaw rotation (planimetric).
- Jitter: scale xy ±5%, z noise ±0.05 m.
- Random dropout of 10–30% of points (simulate occlusion).
- Mixup **across chips** only within same split (optional, v1.1).

### 5.4 Storage layout

```
data/
  raw/laz/                    # downloaded tiles (gitignored)
  manifests/
    gent_trees.parquet
    gent_laz_tiles.csv
    split_meta.json
  processed/
    chips/{chip_id}.npz       # float32 xyz, features, heatmap, mask
  exports/
    inference/{chip_id}.json.gz
```

`npz` schema (v1):

| Array | Shape | Description |
|-------|-------|-------------|
| `xyz` | N×3 | Lambert xy + h_agl |
| `feat` | N×F | intensity, return ratio, optional rgb |
| `heatmap` | H×W | float32 target |
| `ignore` | H×W | uint8 |
| `qc` | scalar attrs in sidecar JSON | `ground_unreliable`, density stats |

### 5.5 Chip sampling (empty chips and class balance)

Urban Gent has far more **unlabeled** surface than inventory points. v1 defaults:

| Mode | Policy |
|------|--------|
| **Training** | Include **positive** chips (≥1 eligible `gis_id` fully inside chip, not in edge ignore band) and **negative** chips (zero eligible trees) |
| **Negative sampling ratio** | **1:2** positive:negative per epoch (draw without replacement within train sheets) |
| **Negative source** | Chips from train kaartbladen with centroid on road/roof-heavy cells (optional GRB road/building mask >50% of chip area) |
| **Val / test** | **All** chips in region — no subsampling (metrics need full FP rate) |

Every eligible stake inside a tile is a positive. There is no edge band that drops it from the loss.

**Alternative (ablation):** train on positive-only chips — faster but higher FP; document if used.

### 5.6 Flight-strip and density controls

- Log `point_source_id` in features.  
- Optional **strip cap:** max 5k points per source ID per chip (random subsample).  
- Optional **voxel subsample** to 8 pts/m² before voxelization ([DATA.md](DATA.md)).

---

## 6. Model architecture (PyTorch)

### 6.1 Design principles

- **Start small** — a model that trains on a single GPU in <2 h on a Gent subset beats a fragile Point Transformer.
- **Prefer 2.5D** for v1 — BEV grid from points is fast and pairs well with heatmap heads (similar to CenterNet / FCOS3D-lite in bird’s-eye view).
- **Keep a point-cloud path** — optional parallel branch for ablation (PointNet++ encoder → max pool → concat to BEV).

### 6.2 Recommended v1 architecture: `GentTreeDetector`

```
Points (N×F)
    → Voxelize (voxel size 0.25 m, max points/voxel 32)
    → Sparse 3D CNN (3–4 layers, channels 32→128)  [torchsparse OR MinkowskiEngine if available; else dense BEV]
    → Collapse Z → BEV tensor (C×H×W)
    → ResNet-style 2D CNN (U-Net decoder)
    → Heads: center heatmap (sigmoid), optional offset (2ch), optional height (1ch)
```

| Component | PyTorch implementation |
|-----------|------------------------|
| Voxelization | **Dense BEV** scatter max/mean (v1, CPU/GPU friendly); sparse libs optional later |
| Loss | Focal loss on heatmap + L1 on offsets at peak locations |
| Optimizer | AdamW, lr 1e-3, cosine decay |
| Batch | 4–8 chips (mixed precision `torch.cuda.amp` when GPU available) |

### 6.3 Baseline B (optional)

Point-wise **RandLA-Net** or **PointNet++** with 3 classes — useful for ablation; not the primary metric unless detection head underperforms CHM baseline.

### 6.4 Reproducibility

- Fixed seeds per split in `split_meta.json`.
- Save `config.yaml` + git SHA + manifest hashes in each checkpoint (`checkpoint["meta"]`).
- Log to local CSV or TensorBoard; Weights & Biases optional.

---

## 7. Training & evaluation workflow

### 7.1 CLI (intended)

```bash
# Prepare chips for Gent train/val/test splits
python -m treelidarnet.data.build_chips --split all

# Train
python -m treelidarnet.train --config configs/gent_bev_v1.yaml

# Evaluate on test split → metrics JSON (see docs/EVAL.md)
python -m treelidarnet.eval --checkpoint runs/gent_bev_v1/best.pt

# Run inference on one chip → Deck.GL export
python -m treelidarnet.infer --checkpoint best.pt --chip-id gent_50m_... \
    --export exports/inference/gent_50m_....json.gz
```

### 7.2 Compute expectations

| Stage | Rough resource |
|-------|----------------|
| Tile download | Manual/CSV bulk; 100s of GB for full Gent — **start with 20–50 tiles** |
| Chip build | CPU-bound; laspy/lazrs + numpy |
| Training v1 | 1× consumer GPU, hours not days on subset; **CPU training** supported (small subset, no AMP) |
| Full-city eval | CPU post-process; minutes |

---

## 8. Visualization & product surface

Full detail: [VISUALIZATION.md](VISUALIZATION.md).

**Summary:**

- **MapLibre** dark basemap + **`deck.MapboxOverlay`** (same integration as HugATree `map.js` and Luwe `trees_in_ghent.html`).
- Layers: ground-truth scatter (emerald), predictions (cyan / orange FP / rose FN halo).
- Optional heatmap toggle; tooltips with `gis_id`, score, match status, link to HugATree slug.

**Inference viewer (v1):** static HTML/JS in `viewer/` loading gzip JSON — no Flask required for model iteration.

---

## 9. Repository layout (target)

```
TreeLidarNet/
  docs/                 # This spec + DATA + VISUALIZATION
  configs/              # YAML hyperparameters
  treelidarnet/
    data/               # download, chip builder, manifests
    models/             # PyTorch modules
    train.py
    eval.py
    infer.py
    metrics.py          # matching, PR curves
  viewer/               # Deck.GL static inference UI
  scripts/              # one-off GDAL/LAZ utilities
  tests/                # pytest: metrics, transforms, manifest integrity
  pyproject.toml        # Python deps: torch, laspy, numpy, pyproj, ...
```

**Reference repos (read-only patterns, optional git submodules later):**

| Repo | Borrow |
|------|--------|
| `~/Dev/HugATree` | Gent import, `tree_map_cache` radius heuristic, map UX |
| `~/Dev/LuweDaman` | Playground loader pattern, panel chrome |

---

## 10. Dependencies

| Area | Packages |
|------|----------|
| Deep learning | **PyTorch** (≥2.3), torchvision optional |
| Point I/O | laspy + lazrs (or pdal Python bindings) |
| Geo | pyproj, shapely, pyogrio/geopandas |
| Numerics | numpy, scipy (Hungarian match), zarr optional |
| Viz export | orjson, gzip |
| Quality | pytest, ruff |

Avoid committing LaZ archives. Use `data/.gitkeep` + documented manifests.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Inventory ≠ LiDAR trees | Distance-based metrics; “discovery” bucket; visual QA |
| 2013–2015 LiDAR vs 2026 inventory | Focus on mature urban trees; expect FN on new plantings |
| Flight strip artifacts | `Point Source ID` feature; strip-aware val tiles |
| Class imbalance (mostly non-tree) | Focal loss; **§5.5** negative chips; GRB hard negatives (v1.1) |
| Private GPU memory | Smaller chips (32 m), voxel size 0.35 m |
| Legal / attribution | [LEGAL.md](LEGAL.md); no raw LAZ in git |

---

## 12. Milestones

### M0 — Bootstrap (week 1)

- [ ] Gent tree import to Parquet (reuse HugATree field mapping).
- [ ] Download LAZ for **one** Gent park + **one** residential street.
- [ ] CHM baseline script + 3 m F1 on tiny manual split.
- [ ] `docs/` accepted (this spec).

### M1 — Data plane (weeks 2–3)

- [ ] Tile manifest for Gent bbox; chip builder to `processed/`.
- [ ] Geographic train/val/test in `split_meta.json`.
- [ ] Unit tests: CRS round-trip, chip ID stability.

### M2 — Model v1 (weeks 3–5)

- [ ] BEV detector trains on subset; TensorBoard curves.
- [ ] `eval.py` → metrics JSON + PR at 2/3/5 m.
- [ ] Beat CHM baseline on val.

### M3 — Inference map (weeks 5–6)

- [ ] `infer.py` → gzip JSON matching VISUALIZATION schema.
- [ ] `viewer/` Deck.GL page with GT + preds + toggles.
- [ ] Side-by-side screenshot parity with Luwe playground chrome.

### M4 — Hardening (ongoing)

- [ ] Expand tiles toward full Gent.
- [ ] Species-stratified error report.
- [ ] Optional HugATree/Luwe embedding of viewer.

---

## 13. Informal research questions

- How much F1 comes from **height threshold** alone in urban Gent?
- Do **false positives** cluster on hedges, utility poles, or churchyard yews?
- Does the model find **plausible trees** far from any `gis_id` (inventory gaps)?
- Can **stamomtrek** improve calibration of predicted crown radius on the map?

---

## 14. Related documents

| Doc | Contents |
|-----|----------|
| [DATA.md](DATA.md) | LiDAR, Gent inventory, splits, manifests, eligibility |
| [EVAL.md](EVAL.md) | Post-processing, NMS, metrics, discoveries |
| [LEGAL.md](LEGAL.md) | Redistribution and attribution |
| [VISUALIZATION.md](VISUALIZATION.md) | Deck.GL layers, JSON schema, UX |
| [HugATree SPEC](../../Dev/HugATree/docs/SPEC.md) | Sister product context |
| [HugATree DATA](../../Dev/HugATree/docs/DATA.md) | Gent field reference |

---

## Appendix A — Frozen CHM baseline (Phase A)

Used for M0 and ongoing comparison. **Do not change** these without bumping `baseline_version` in eval JSON.

| Parameter | Value |
|-----------|--------|
| `baseline_version` | `chm_v1` |
| DSM source | DHMV II **DSM 1 m** (official download.vlaanderen.be product) |
| DTM source | DHMV II **DTM 1 m** (same) |
| CHM | `DSM - DTM` (per cell); clamp min 0 |
| Min canopy height | **3.0 m** above ground |
| Peak filter | `scipy.ndimage.maximum_filter` window **5×5** cells (5 m) |
| Min peak CHM | **3.0 m** (same as threshold) |
| Peak spacing NMS | **3.0 m** between CHM peaks (2D greedy) |
| Building mask | **None** in v1 — expect roof FP; note in reports |

Peaks → Lambert xy → same city merge + match as neural model ([EVAL.md](EVAL.md)).

**Literature pointers (external baselines):** R `lidR` tree detection tutorials; watershed on CHM common in forestry — cite if reproducing published params.

---

## 15. Naming note

Official Flanders branding is **DHMV II** (*Digitaal Hoogtemodel Vlaanderen II*). “DHV II” appears colloquially; all download links and credits should use **DHMV II** and **EODaS OpenLiDAR**.
