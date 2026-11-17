# Task: Find trees in Ghent broadly (canopy v2)

**Primary product:** citywide **LiDAR tree candidates** (peaks + score) from DHMV II, not a replacement for Stad Gent’s register.

**Secondary:** strict / audit match vs [locaties-bomen-gent](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) for sanity checks.

---

## Training (soft register)

| Signal | Role |
|--------|------|
| Register peaks | **Positive** heatmap Gaussians at eligible `gis_id` points (weak, not exhaustive); **σ scales with `stamomtrek`** (1.0–2.5 m) so large crowns get wider targets |
| **Unknown mask** | Pixels > `unknown_band_m` from any register point → **no negative loss** (unlisted canopy allowed) |
| **Hard negative** | GRB `IngeschetstGebouw` + `Wegsegment` (2 m buffer) → strong negative (`data-download-grb`, `data-patch-chips-grb`) |
| Chip edges | Ignore mask is clear: every stake in the tile is a positive |

Config: [configs/gent_canopy_v2.yaml](../configs/gent_canopy_v2.yaml).

Fine-tune from `runs/gent_real_v1/best.pt` with `--resume` when weights exist.

---

## Chip coverage

| Set | CLI | Purpose |
|-----|-----|---------|
| Register-skewed | `data-build-gent-chips` | ~10k chips with ≥1 eligible tree |
| Full bbox grid | `data-build-gent-grid` (+ `--skip-existing`) | 50 m grid over Gent bbox ∩ LAZ |
| In-place target refresh | `data-patch-chips-heatmap` | Rewrite `heatmap` (and optionally GRB) in existing `.npz` without rebuilding from LAZ |

Index: `data/manifests/chips_index_full.parquet` (not committed). Same npz dir: `data/processed/chips/`.

**Ops:** If you change heatmap σ after a partial grid build, **stop** the grid, run `data-patch-chips-heatmap` on all current chips, then resume grid with `--skip-existing`.

---

## Evaluation hierarchy

| Tier | Metrics |
|------|---------|
| **Primary** | Audit F1 @ 3 m, discovery count, `pred_per_km2` |
| **Secondary** | Strict F1 @ 3 m vs register |
| **Sanity** | CHM baseline on held-out sheets |

See [EVAL.md](EVAL.md), [LABELS.md](LABELS.md).

---

## Legal

Discoveries and canopy maps are **research / exploration** outputs — not official municipal data ([LEGAL.md](LEGAL.md)).

---

## Second stage (crop verifier)

BEV `epoch3_hot` stays frozen for proposals. A **PointNet++** model on ~512-point LiDAR crops filters each detection (tree vs not-tree). Train on **every eligible register stake** plus GRB / `hard_negative` negatives—not carpet or off-register mid scores. See [SPEC.md](SPEC.md) §6.5 and [THOUGHT_PROCESS.MD](../THOUGHT_PROCESS.MD) §37.

---

## Out of scope (v2.0)

- DHMV-era RGB / satellite fusion (v2.1)
- OSM hard negatives (v2.1)
- Flanders-wide training
