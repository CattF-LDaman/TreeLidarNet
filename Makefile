.PHONY: sync sync-cpu test lint tensorboard viewer-serve viewer-crop-qa patch-chips-rgb-full help

help:
	@echo "Targets: sync, sync-cpu, test, lint, tensorboard, viewer-crop-qa, patch-chips-rgb-full, filter-export-verifier-full"

patch-chips-rgb-full:
	PYTHONUNBUFFERED=1 uv run treelidarnet data-patch-chips-rgb \
	  --all-chips --only-missing-rgb --refresh-missing-manifest \
	  --jobs 1

tensorboard:
	uv run tensorboard --logdir runs --bind_all

viewer-serve:
	python -m http.server 8765 --bind 127.0.0.1

viewer-crop-qa:
	uv run treelidarnet serve-crop-qa \
	  --verifier-v2 runs/gent_crop_verifier_v2/best.pt \
	  --verifier-v3 runs/gent_crop_verifier_v3_rgb/best.pt

infer-gent-full:
	PYTHONUNBUFFERED=1 uv run treelidarnet infer-gent \
	  --checkpoint runs/gent_canopy_v2/best.pt \
	  --chips-index data/manifests/chips_index_full.parquet \
	  --score-threshold 0.2 \
	  --export exports/inference/gent_city_gent_canopy_v2.json.gz \
	  --batch-size 8 --num-workers 4 --progress-every 200

infer-gent-full-live:
	PYTHONUNBUFFERED=1 uv run treelidarnet infer-gent --live \
	  --checkpoint runs/gent_canopy_v2/best.pt \
	  --chips-index data/manifests/chips_index_full.parquet \
	  --score-threshold 0.2 \
	  --export exports/inference/gent_city_gent_canopy_v2_full.json.gz \
	  --progress-every 200 --live-flush-every 400

filter-export-live-center:
	uv run treelidarnet filter-export --live \
	  --input exports/inference/gent_city_gent_canopy_v2.json.gz \
	  --output exports/inference/gent_city_gent_canopy_v2_verified.json.gz \
	  --verifier runs/gent_crop_verifier_v2/best.pt \
	  --bev-min 0.9 --limit 8000

filter-export-live-compare:
	PYTHONUNBUFFERED=1 uv run treelidarnet filter-export --live \
	  --input exports/inference/gent_city_gent_canopy_v2.json.gz \
	  --output exports/inference/gent_city_gent_canopy_v2_verified_v2_v3.json.gz \
	  --verifier-v2 runs/gent_crop_verifier_v2/best.pt \
	  --verifier-v3 runs/gent_crop_verifier_v3_rgb/best.pt \
	  --bev-min 0.85 --limit 25000

filter-export-verifier-full:
	PYTHONUNBUFFERED=1 uv run treelidarnet filter-export --live \
	  --input exports/inference/gent_city_gent_canopy_v2.json.gz \
	  --output exports/inference/gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz \
	  --verifier-v2 runs/gent_crop_verifier_v2/best.pt \
	  --verifier-v3 runs/gent_crop_verifier_v3_rgb/best.pt \
	  --bev-min 0.85

sync:
	uv sync

sync-cpu:
	uv sync --no-default-groups --group dev --group cpu

test:
	uv run pytest -q

lint:
	uv run ruff check treelidarnet tests
