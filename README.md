# TreeLidarNet

Find **trees in Ghent broadly** from **DHMV II LiDAR** (PyTorch BEV detector). Stad Gent’s [**locaties-bomen-gent**](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) is **weak supervision** and a **secondary** benchmark — not the product census. See [docs/TASK_GHENT_BROAD.md](docs/TASK_GHENT_BROAD.md) and [docs/LABELS.md](docs/LABELS.md).

## Quick start (uv)

```bash
cd TreeLidarNet
uv sync
make test
uv run treelidarnet data-import-gent
uv run treelidarnet data-write-split-meta
```

See [docs/DEV.md](docs/DEV.md) for CUDA/CPU notes, LAZ chip build, train/infer, and the Deck.GL viewer.

**Documentation:**

| Doc | Contents |
|-----|----------|
| [docs/TASK_GHENT_BROAD.md](docs/TASK_GHENT_BROAD.md) | Canopy v2 task: soft register, unknown mask, grid chips |
| [docs/SPEC.md](docs/SPEC.md) | Goals, ML problem definition, architecture, milestones |
| [docs/LABELS.md](docs/LABELS.md) | Register as weak labels; audit primary, strict secondary |
| [docs/DATA.md](docs/DATA.md) | LiDAR tiles, labels, CRS, splits, eligibility, versioning |
| [docs/EVAL.md](docs/EVAL.md) | Peak extract, NMS, tree-level metrics, discoveries |
| [docs/LEGAL.md](docs/LEGAL.md) | Attribution, redistribution, model card |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | Deck.GL + MapLibre inference UI |
| [docs/DEV.md](docs/DEV.md) | uv workflow and Fedora tips |

**Reference projects (sibling repos):**

- [`~/Dev/HugATree`](../Dev/HugATree) — production map UX, scatter cache shape, `gis_id` labels
- [`~/Dev/LuweDaman`](../Dev/LuweDaman) — playground “Trees in Ghent” Deck.GL pattern
