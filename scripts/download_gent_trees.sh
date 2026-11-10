#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run treelidarnet data-import-gent --out data/manifests/gent_trees.parquet
