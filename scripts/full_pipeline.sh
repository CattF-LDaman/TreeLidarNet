#!/usr/bin/env bash
# Gent: LAZ → chips → train (run from repo root; LAZ step takes hours).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/laz data/logs runs

echo "== Gent trees (skip if parquet fresh) =="
uv run treelidarnet data-import-gent

echo "== DHMV II LAZ for municipal bbox (~1330 tiles) =="
uv run treelidarnet data-download-laz --workers 8 2>&1 | tee -a data/logs/laz_download.log

echo "== Chips for eligible trees (~10k) =="
uv run treelidarnet data-build-gent-chips 2>&1 | tee -a data/logs/build_chips.log

echo "== Train (TensorBoard: make tensorboard) =="
uv run treelidarnet train --config configs/gent_bev_v1.yaml --run-name gent_real_v1
