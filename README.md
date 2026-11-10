# TreeLidarNet

Train PyTorch models to **align DHMV II LiDAR with Stad Gent’s municipal tree register** ([**locaties-bomen-gent**](https://data.stad.gent/explore/dataset/locaties-bomen-gent/) — managed trees only, not every canopy). See [docs/LABELS.md](docs/LABELS.md) for partial-inventory training and **strict vs audit** metrics.

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
| [docs/SPEC.md](docs/SPEC.md) | Goals, ML problem definition, architecture, milestones |
| [docs/LABELS.md](docs/LABELS.md) | Partial inventory, training objective, strict vs audit eval |
| [docs/DATA.md](docs/DATA.md) | LiDAR tiles, labels, CRS, splits, eligibility, versioning |
| [docs/EVAL.md](docs/EVAL.md) | Peak extract, NMS, tree-level metrics, discoveries |
| [docs/LEGAL.md](docs/LEGAL.md) | Attribution, redistribution, model card |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | Deck.GL + MapLibre inference UI |
| [docs/DEV.md](docs/DEV.md) | uv workflow and Fedora tips |

**Reference projects (sibling repos):**

- [`~/Dev/HugATree`](../Dev/HugATree) — production map UX, scatter cache shape, `gis_id` labels
- [`~/Dev/LuweDaman`](../Dev/LuweDaman) — playground “Trees in Ghent” Deck.GL pattern
