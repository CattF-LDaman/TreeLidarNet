# TreeLidarNet — developer setup (uv)

## Requirements

- [uv](https://docs.astral.sh/uv/) (latest)
- Python **3.12** (see `.python-version`)
- Fedora: `sudo dnf install gdal gdal-devel` if `rasterio` fails to import

## Sync environment

```bash
cd /home/catt/TreeLidarNet
uv sync              # CUDA PyTorch (cu124 index) + dev deps
make test
```

**CPU-only PyTorch** (no NVIDIA):

```bash
uv sync --no-default-groups --group dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If cu124 wheels fail on your driver, edit `pyproject.toml` to use `cu121` index or use the CPU path above.

## Common commands

```bash
uv run treelidarnet data-import-gent
uv run treelidarnet data-write-split-meta
uv run treelidarnet data-build-chips --laz-dir data/raw/laz --origins "200000,193000"
uv run treelidarnet train --config configs/gent_bev_v1.yaml

# Canopy v2 (broad tree finding, soft register + unknown mask):
# uv run treelidarnet data-download-grb
# uv run treelidarnet data-patch-chips-grb          # optional: GRB hard_negative on existing npz
# uv run treelidarnet data-patch-chips-heatmap      # rewrite heatmap (variable σ); refreshes GRB unless --no-grb
# uv run treelidarnet data-build-gent-grid --laz-dir data/raw/laz --skip-existing   # resume full grid (~days)
# uv run treelidarnet train --config configs/gent_canopy_v2.yaml    # loads gent_real_v1/best.pt via init_checkpoint
# uv run treelidarnet train --config configs/gent_canopy_v2.yaml --resume

# Checkpoints: `runs/<run_name>/latest.pt` (every 500 steps + each epoch + on Ctrl+C),
# `best.pt` (lowest epoch-mean train loss). Resume:
# uv run treelidarnet train --config configs/gent_bev_v1.yaml --run-name gent_real_v1 --resume

# In another terminal while training (logs under runs/<run_name>/tb):
make tensorboard
# Open http://localhost:6006
uv run treelidarnet infer --checkpoint runs/gent_bev_v1/best.pt --chip-id gent_50m_200000_193000
uv run ruff check treelidarnet tests
```

## Data layout

- `data/raw/laz/` — downloaded LAZ (gitignored)
- `data/processed/chips/` — training npz (gitignored; do not commit). After **unknown-mask** or **heatmap σ** logic changes: use `data-patch-chips-heatmap` on existing chips (no LAZ re-read) or full rebuild; do not run grid build and heatmap patch on the same chips at once.
- `data/logs/` — long jobs: `build_gent_grid.log`, `patch_heatmap.log` (optional `tee` / `nohup`)
- `data/manifests/chips_index_full.parquet` — full 50 m grid manifest from `data-build-gent-grid`
- `data/manifests/split_meta.json` — edit kaartblad lists after `data-write-split-meta`

## Viewer

City demo (after `infer-gent`):

```bash
uv run treelidarnet infer-gent --checkpoint runs/gent_canopy_v2/best.pt --score-threshold 0.2
make viewer-serve   # http://127.0.0.1:8765/viewer/ghent.html
```

Live map while inferring (polls every 2s; flush every 150 chips):

```bash
uv run treelidarnet infer-gent --live --checkpoint runs/gent_real_v1/best.pt
# http://127.0.0.1:8765/viewer/ghent_live.html
```

Chip QA: open `viewer/index.html` and load `exports/inference/*.json.gz`.
