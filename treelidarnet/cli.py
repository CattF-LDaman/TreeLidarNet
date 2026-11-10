"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from treelidarnet.baseline.chm import chm_peaks_for_paths
from treelidarnet.data.build_chips import build_chips_for_origins, discover_laz_dir
from treelidarnet.data.build_gent import build_gent_chips
from treelidarnet.data.build_gent_grid import build_gent_grid_chips
from treelidarnet.data.download_grb import download_gent_grb
from treelidarnet.data.download_laz import download_gent_laz
from treelidarnet.data.import_gent import DEFAULT_EXPORT_URL, import_gent_geojson
from treelidarnet.data.patch_grb_masks import patch_chips_grb
from treelidarnet.data.patch_heatmaps import patch_chips_heatmaps
from treelidarnet.data.split import (
    init_split_meta_from_example,
    load_split_meta,
    update_gent_hashes,
    write_split_meta,
)
from treelidarnet.eval import eval_detections, load_ground_truths
from treelidarnet.infer import infer_chip
from treelidarnet.infer_city import infer_gent_city
from treelidarnet.paths import manifests_dir, repo_root
from treelidarnet.train import train_from_config


def _cmd_data_import_gent(args: argparse.Namespace) -> int:
    out = Path(args.out)
    meta = import_gent_geojson(url=args.url or DEFAULT_EXPORT_URL, out=out)
    print(json.dumps(meta, indent=2))
    split_path = manifests_dir() / "split_meta.json"
    if split_path.is_file():
        sm = load_split_meta(split_path)
        sm = update_gent_hashes(
            sm,
            sha256=meta["sha256"],
            record_count=meta["record_count"],
            eligible_count=meta["eligible_count"],
        )
        write_split_meta(split_path, sm)
    return 0


def _cmd_data_write_split_meta(args: argparse.Namespace) -> int:
    example = Path(args.example)
    dest = Path(args.dest)
    init_split_meta_from_example(example, dest)
    print(f"Wrote {dest} — edit sheets before build-chips.")
    return 0


def _cmd_data_download_laz(args: argparse.Namespace) -> int:
    manifest = download_gent_laz(
        trees_parquet=Path(args.trees),
        dest_dir=Path(args.dest),
        max_workers=int(args.workers),
        limit=int(args.limit) if args.limit else None,
    )
    print(f"Manifest: {manifest}")
    return 0


def _cmd_data_build_gent_chips(args: argparse.Namespace) -> int:
    dtm = Path(args.dtm) if args.dtm else None
    grb_dir = Path(args.grb_dir) if args.grb_dir else None
    build_gent_chips(
        trees_parquet=Path(args.trees),
        laz_dir=Path(args.laz_dir),
        dtm_path=dtm,
        limit=int(args.limit) if args.limit else None,
        grb_dir=grb_dir,
        road_buffer_m=float(args.road_buffer_m),
    )
    return 0


def _cmd_data_download_grb(args: argparse.Namespace) -> int:
    meta = download_gent_grb(
        trees_parquet=Path(args.trees),
        dest_dir=Path(args.dest),
        buffer_m=float(args.buffer_m),
    )
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_data_patch_chips_grb(args: argparse.Namespace) -> int:
    patch_chips_grb(
        grb_dir=Path(args.grb_dir),
        index_parquet=Path(args.index) if args.index else None,
        road_buffer_m=float(args.road_buffer_m),
        limit=int(args.limit) if args.limit else None,
    )
    return 0


def _cmd_data_build_gent_grid(args: argparse.Namespace) -> int:
    dtm = Path(args.dtm) if args.dtm else None
    build_gent_grid_chips(
        trees_parquet=Path(args.trees),
        laz_dir=Path(args.laz_dir),
        dtm_path=dtm,
        limit=int(args.limit) if args.limit else None,
        skip_existing=bool(args.skip_existing),
    )
    return 0


def _cmd_data_patch_chips_heatmap(args: argparse.Namespace) -> int:
    patch_chips_heatmaps(
        trees_parquet=Path(args.trees),
        grb_dir=Path(args.grb_dir) if args.grb_dir else None,
        road_buffer_m=float(args.road_buffer_m),
        refresh_hard_negative=not args.no_grb,
        limit=int(args.limit) if args.limit else None,
    )
    return 0


def _cmd_data_build_chips(args: argparse.Namespace) -> int:
    laz_dir = Path(args.laz_dir)
    trees = Path(args.trees)
    dtm = Path(args.dtm) if args.dtm else None
    origins = []
    for part in args.origins.split(";"):
        part = part.strip()
        if not part:
            continue
        x0, y0 = (float(v) for v in part.split(","))
        origins.append((x0, y0))
    laz_paths = discover_laz_dir(laz_dir)
    if not laz_paths:
        raise SystemExit(f"No LAZ files in {laz_dir}")
    idx = build_chips_for_origins(origins, laz_paths=laz_paths, trees_parquet=trees, dtm_path=dtm)
    print(f"Built {len(idx)} chips")
    return 0


def _cmd_baseline_chm(args: argparse.Namespace) -> int:
    peaks = chm_peaks_for_paths(Path(args.dsm), Path(args.dtm))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_rows = [{"x": p.x, "y": p.y, "score": p.score} for p in peaks]
    payload = {"baseline_version": "chm_v1", "predictions": pred_rows}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(peaks)} peaks to {out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    ckpt = train_from_config(
        Path(args.config),
        run_name=args.run_name,
        tensorboard=not args.no_tensorboard,
        resume=args.resume,
    )
    print(f"Checkpoint: {ckpt}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from treelidarnet.postprocess import Detection

    pred_data = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    preds = [
        Detection(x=p["x"], y=p["y"], score=p.get("score", 1.0))
        for p in pred_data["predictions"]
    ]
    gts = load_ground_truths(Path(args.trees))
    discoveries = Path(args.export_discoveries) if args.export_discoveries else None
    out = eval_detections(
        preds,
        gts,
        run_id=args.run_id,
        split=args.split,
        task=args.task,
        export_discoveries=discoveries,
    )
    print(f"Metrics: {out}")
    if discoveries:
        print(f"Discoveries GeoJSON: {discoveries}")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    out = infer_chip(
        checkpoint=Path(args.checkpoint),
        chip_id=args.chip_id,
        trees_parquet=Path(args.trees),
        export_path=Path(args.export) if args.export else None,
        model_id=args.model_id,
        score_threshold=float(args.score_threshold),
    )
    print(f"Export: {out}")
    return 0


def _cmd_infer_gent(args: argparse.Namespace) -> int:
    chips_index = Path(args.chips_index) if args.chips_index else None
    if chips_index is not None and not chips_index.is_file():
        chips_index = None
    out = infer_gent_city(
        checkpoint=Path(args.checkpoint),
        trees_parquet=Path(args.trees),
        export_path=Path(args.export) if args.export else None,
        model_id=args.model_id,
        batch_size=int(args.batch_size),
        limit=int(args.limit) if args.limit else None,
        chips_index=chips_index,
        live=args.live,
        live_flush_every=int(args.live_flush_every),
        progress_every=int(args.progress_every),
        score_threshold=float(args.score_threshold),
    )
    print(f"Export: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="treelidarnet")
    sub = parser.add_subparsers(dest="command", required=True)

    p_imp = sub.add_parser("data-import-gent", help="Fetch Gent GeoJSON → parquet")
    p_imp.add_argument("--url", default=None)
    p_imp.add_argument("--out", default=str(manifests_dir() / "gent_trees.parquet"))
    p_imp.set_defaults(func=_cmd_data_import_gent)

    p_split = sub.add_parser("data-write-split-meta", help="Copy example split_meta.json")
    p_split.add_argument(
        "--example",
        default=str(repo_root() / "data/manifests/split_meta.example.json"),
    )
    p_split.add_argument("--dest", default=str(manifests_dir() / "split_meta.json"))
    p_split.set_defaults(func=_cmd_data_write_split_meta)

    p_dl = sub.add_parser("data-download-laz", help="Download DHMV II LAZ for Gent bbox")
    p_dl.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_dl.add_argument("--dest", default="data/raw/laz")
    p_dl.add_argument("--workers", default="6")
    p_dl.add_argument("--limit", default=None, help="Max 500m tiles (testing)")
    p_dl.set_defaults(func=_cmd_data_download_laz)

    p_bg = sub.add_parser("data-build-gent-chips", help="Build npz chips for all eligible trees")
    p_bg.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_bg.add_argument("--laz-dir", default="data/raw/laz")
    p_bg.add_argument("--dtm", default=None)
    p_bg.add_argument("--limit", default=None)
    p_bg.add_argument("--grb-dir", default=None, help="data/raw/grb if GeoJSON present")
    p_bg.add_argument("--road-buffer-m", default="2.0")
    p_bg.set_defaults(func=_cmd_data_build_gent_chips)

    p_grb_dl = sub.add_parser("data-download-grb", help="GRB buildings+roads GeoJSON for Gent bbox")
    p_grb_dl.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_grb_dl.add_argument("--dest", default="data/raw/grb")
    p_grb_dl.add_argument("--buffer-m", default="500")
    p_grb_dl.set_defaults(func=_cmd_data_download_grb)

    p_grb_patch = sub.add_parser(
        "data-patch-chips-grb",
        help="Rewrite hard_negative in existing chips from GRB GeoJSON",
    )
    p_grb_patch.add_argument("--grb-dir", default="data/raw/grb")
    p_grb_patch.add_argument("--index", default=None)
    p_grb_patch.add_argument("--road-buffer-m", default="2.0")
    p_grb_patch.add_argument("--limit", default=None)
    p_grb_patch.set_defaults(func=_cmd_data_patch_chips_grb)

    p_hm_patch = sub.add_parser(
        "data-patch-chips-heatmap",
        help="Rewrite heatmap (variable σ) in existing npz; optional GRB refresh",
    )
    p_hm_patch.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_hm_patch.add_argument("--grb-dir", default="data/raw/grb")
    p_hm_patch.add_argument("--road-buffer-m", default="2.0")
    p_hm_patch.add_argument("--no-grb", action="store_true")
    p_hm_patch.add_argument("--limit", default=None)
    p_hm_patch.set_defaults(func=_cmd_data_patch_chips_heatmap)

    p_grid = sub.add_parser(
        "data-build-gent-grid",
        help="50 m grid chips over Gent bbox → chips_index_full.parquet",
    )
    p_grid.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_grid.add_argument("--laz-dir", default="data/raw/laz")
    p_grid.add_argument("--dtm", default=None)
    p_grid.add_argument("--limit", default=None)
    p_grid.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rebuild npz that already exist (resume grid after patch)",
    )
    p_grid.set_defaults(func=_cmd_data_build_gent_grid)

    p_chips = sub.add_parser("data-build-chips", help="Build npz chips from LAZ")
    p_chips.add_argument("--laz-dir", required=True)
    p_chips.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_chips.add_argument("--dtm", default=None)
    p_chips.add_argument(
        "--origins",
        required=True,
        help="Lambert origins x0,y0 separated by ; e.g. '100000,195000;100050,195000'",
    )
    p_chips.set_defaults(func=_cmd_data_build_chips)

    p_chm = sub.add_parser("baseline-chm", help="CHM peak baseline")
    p_chm.add_argument("--dsm", required=True)
    p_chm.add_argument("--dtm", required=True)
    p_chm.add_argument("--out", default="exports/baseline/chm_peaks.json")
    p_chm.set_defaults(func=_cmd_baseline_chm)

    p_train = sub.add_parser("train", help="Train BEV detector")
    p_train.add_argument("--config", default="configs/gent_bev_v1.yaml")
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard scalar/image logging",
    )
    p_train.add_argument(
        "--resume",
        action="store_true",
        help="Continue from runs/<run_name>/latest.pt (or best.pt)",
    )
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate prediction JSON")
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_eval.add_argument("--run-id", default="eval_run")
    p_eval.add_argument("--split", default="test")
    p_eval.add_argument("--task", default="ghent_canopy_detection")
    p_eval.add_argument(
        "--export-discoveries",
        default=None,
        help="Optional GeoJSON path (e.g. exports/discoveries_run.geojson)",
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_infer = sub.add_parser("infer", help="Infer one chip")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--chip-id", required=True)
    p_infer.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_infer.add_argument("--export", default=None)
    p_infer.add_argument("--model-id", default="gent_bev_v1")
    p_infer.add_argument(
        "--score-threshold",
        default="0.2",
        help="Minimum model confidence (0–1) to export a peak",
    )
    p_infer.set_defaults(func=_cmd_infer)

    p_city = sub.add_parser("infer-gent", help="Infer all chips → city map JSON.gz")
    p_city.add_argument(
        "--checkpoint",
        default=str(repo_root() / "runs/gent_canopy_v2/best.pt"),
    )
    p_city.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_city.add_argument("--export", default=None)
    p_city.add_argument("--model-id", default="gent_canopy_v2")
    p_city.add_argument(
        "--chips-index",
        default=str(manifests_dir() / "chips_index_full.parquet"),
        help="Manifest parquet; omit file to use all *.npz in chips dir",
    )
    p_city.add_argument("--batch-size", default="8")
    p_city.add_argument("--limit", default=None, help="Max chips (smoke test)")
    p_city.add_argument(
        "--live",
        action="store_true",
        help="Write exports/inference/live/ for ghent_live.html polling",
    )
    p_city.add_argument("--live-flush-every", default="150")
    p_city.add_argument("--progress-every", default="50")
    p_city.add_argument(
        "--score-threshold",
        default="0.2",
        help="Minimum model confidence (0–1) to export a peak",
    )
    p_city.set_defaults(func=_cmd_infer_gent)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
