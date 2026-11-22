.PHONY: sync sync-cpu test lint tensorboard viewer-serve viewer-stop viewer-crop-qa explorer-bundle hugatree-bundle explorer-serve combo-export-gif patch-chips-rgb-full help

VIEWER_PORT ?= 8765

help:
	@echo "Targets: sync, sync-cpu, test, lint, tensorboard, viewer-crop-qa, explorer-bundle, hugatree-bundle, explorer-serve, combo-export-gif, patch-chips-rgb-full, filter-export-verifier-full"

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

explorer-bundle:
	PYTHONUNBUFFERED=1 uv run treelidarnet data-build-explorer-bundle

hugatree-bundle:
	PYTHONUNBUFFERED=1 uv run treelidarnet data-build-hugatree-bundle \
	  --output /home/catt/Dev/HugATree/static/data/lidar

hugatree-bundle-monitor: ## Watch LiDAR pack growth (safe while build runs)
	@INTERVAL=30 LOG_PROGRESS=1 ./scripts/monitor_hugatree_bundle.sh /home/catt/Dev/HugATree/static/data/lidar

combo-export-gif:
	PYTHONUNBUFFERED=1 uv run treelidarnet data-render-combo-gif

viewer-stop:
	@PID=$$(ss -H -tlnp 'sport = :$(VIEWER_PORT)' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1); \
	if [ -n "$$PID" ]; then \
	  echo "Stopping process $$PID on port $(VIEWER_PORT)"; \
	  kill $$PID 2>/dev/null || true; \
	  sleep 0.4; \
	fi

explorer-serve: viewer-stop
	uv run treelidarnet serve-explorer --port $(VIEWER_PORT)

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
	  --combine-mode weighted --verifier-threshold 0.45 \
	  --bev-min 0.85 --limit 25000

filter-export-verifier-full:
	PYTHONUNBUFFERED=1 uv run treelidarnet filter-export --live \
	  --input exports/inference/gent_city_gent_canopy_v2.json.gz \
	  --output exports/inference/gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz \
	  --verifier-v2 runs/gent_crop_verifier_v2_fp2/best.pt \
	  --verifier-v3 runs/gent_crop_verifier_v3_rgb_fp2/best.pt \
	  --combine-mode weighted --verifier-threshold 0.45 \
	  --bev-min 0.85 --infer-batch-size 64

sync:
	uv sync

sync-cpu:
	uv sync --no-default-groups --group dev --group cpu

test:
	uv run pytest -q

lint:
	uv run ruff check treelidarnet tests
