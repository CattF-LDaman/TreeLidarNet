.PHONY: sync sync-cpu test lint tensorboard viewer-serve help

help:
	@echo "Targets: sync, sync-cpu, test, lint, tensorboard"

tensorboard:
	uv run tensorboard --logdir runs --bind_all

viewer-serve:
	python -m http.server 8765 --bind 127.0.0.1

sync:
	uv sync

sync-cpu:
	uv sync --no-default-groups --group dev --group cpu

test:
	uv run pytest -q

lint:
	uv run ruff check treelidarnet tests
