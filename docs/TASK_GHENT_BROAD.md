# Task: Find trees in Ghent broadly (canopy v2)

**Primary product:** citywide **LiDAR tree candidates** (peaks + score) from DHMV II, not a replacement for Stad Gent’s register.

**Secondary:** strict / audit match vs [locaties-bomen-gent](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) for sanity checks.

---

## Training (soft register)

| Signal | Role |
|--------|------|
| Register peaks | **Positive** heatmap Gaussians at eligible `gis_id` points (weak, not exhaustive) |
| **Unknown mask** | Pixels > `unknown_band_m` from any register point → **no negative loss** (unlisted canopy allowed) |
| **Hard negative** | GRB `IngeschetstGebouw` + `Wegsegment` (2 m buffer) → strong negative (`data-download-grb`, `data-patch-chips-grb`) |
| Chip edges | Existing ignore annulus (unchanged) |

Config: [configs/gent_canopy_v2.yaml](../configs/gent_canopy_v2.yaml).

Fine-tune from `runs/gent_real_v1/best.pt` with `--resume` when weights exist.

---

## Chip coverage

| Set | CLI | Purpose |
|-----|-----|---------|
| Register-skewed | `data-build-gent-chips` | ~10k chips with ≥1 eligible tree |
| Full bbox grid | `data-build-gent-grid` | 50 m grid over Gent bbox ∩ LAZ |

Index: `data/manifests/chips_index_full.parquet` (not committed). Same npz dir: `data/processed/chips/`.

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

## Out of scope (v2.0)

- DHMV-era RGB / satellite fusion (v2.1)
- OSM hard negatives (v2.1)
- Flanders-wide training
