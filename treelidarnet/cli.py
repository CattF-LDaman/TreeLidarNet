"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from treelidarnet.baseline.chm import chm_peaks_for_paths
from treelidarnet.data.build_chips import build_chips_for_origins, discover_laz_dir
from treelidarnet.data.build_gent import build_gent_chips
from treelidarnet.data.build_gent_grid import build_gent_grid_chips, rebuild_empty_chips
from treelidarnet.data.download_grb import download_gent_grb
from treelidarnet.data.download_laz import download_gent_laz, download_missing_strips
from treelidarnet.data.import_gent import DEFAULT_EXPORT_URL, import_gent_geojson
from treelidarnet.data.patch_chip_rgb import (
    default_crops_index,
    default_dtm_path,
    patch_chips_rgb,
)
from treelidarnet.data.patch_grb_masks import patch_chips_grb
from treelidarnet.data.patch_heatmaps import (
    patch_chips_clear_ignore,
    patch_chips_heatmaps,
    patch_chips_unknown_band,
)
from treelidarnet.data.split import (
    init_split_meta_from_example,
    load_split_meta,
    update_gent_hashes,
    write_split_meta,
)
from treelidarnet.eval import eval_detections, load_ground_truths
from treelidarnet.infer import infer_chip
from treelidarnet.infer_city import infer_gent_city
from treelidarnet.paths import exports_dir, manifests_dir, repo_root
from treelidarnet.qa_center import compare_center_checkpoints, promote_or_revert
from treelidarnet.data.build_crop_index import build_crop_index
from treelidarnet.train import train_from_config
from treelidarnet.train_crop import train_crop_from_config
from treelidarnet.crop_qa_server import serve_crop_qa
from treelidarnet.explorer.app import serve_explorer
from treelidarnet.explorer.bundle import cli_build_bundle
from treelidarnet.explorer.eval_flags import eval_explorer_flags, print_flag_eval_report
from treelidarnet.verify_export import filter_export


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
    kwargs = {
        "trees_parquet": Path(args.trees),
        "dest_dir": Path(args.dest),
        "max_workers": int(args.workers),
        "limit": int(args.limit) if args.limit else None,
    }
    if args.strips:
        manifest = download_missing_strips(**kwargs)
    else:
        manifest = download_gent_laz(**kwargs)
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


def _cmd_data_rebuild_empty_chips(args: argparse.Namespace) -> int:
    dtm = Path(args.dtm) if args.dtm else None
    grb_dir = Path(args.grb_dir) if args.grb_dir else None
    rebuild_empty_chips(
        trees_parquet=Path(args.trees),
        laz_dir=Path(args.laz_dir),
        index_parquet=Path(args.index) if args.index else None,
        dtm_path=dtm,
        grb_dir=grb_dir,
        road_buffer_m=float(args.road_buffer_m),
        limit=int(args.limit) if args.limit else None,
    )
    return 0


def _cmd_data_patch_chips_ignore(args: argparse.Namespace) -> int:
    index = Path(args.index) if args.index else manifests_dir() / "chips_index_full.parquet"
    patch_chips_clear_ignore(
        index_parquet=index,
        limit=int(args.limit) if args.limit else None,
    )
    return 0


def _cmd_data_patch_chips_unknown(args: argparse.Namespace) -> int:
    index = Path(args.index) if args.index else manifests_dir() / "chips_index_full.parquet"
    patch_chips_unknown_band(
        trees_parquet=Path(args.trees),
        index_parquet=index,
        unknown_band_m=float(args.band_m),
        limit=int(args.limit) if args.limit else None,
    )
    return 0


def _cmd_data_patch_chips_rgb(args: argparse.Namespace) -> int:
    dtm = Path(args.dtm) if args.dtm else default_dtm_path()
    crops = Path(args.crops_index) if args.crops_index else default_crops_index()
    chips_index = Path(args.chips_index) if args.chips_index else None
    use_all = bool(args.all_chips)
    patch_chips_rgb(
        laz_dir=Path(args.laz_dir),
        dtm_path=dtm,
        crops_index=None if use_all or chips_index else crops,
        chips_index=chips_index,
        all_chips=use_all,
        limit=int(args.limit) if args.limit else None,
        skip_existing=not args.force,
        jobs=int(args.jobs),
        offset=int(args.offset),
        only_missing_rgb=bool(args.only_missing_rgb),
        refresh_missing_manifest=bool(args.refresh_missing_manifest),
        register_chips=bool(args.register_chips),
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


def _cmd_eval_explorer_flags(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else exports_dir() / "qa" / "explorer_fp_rescore.parquet"
    df, summary = eval_explorer_flags(
        flags_path=Path(args.flags) if args.flags else None,
        v2_baseline=Path(args.verifier_v2_baseline) if args.verifier_v2_baseline else None,
        v2_fp=Path(args.verifier_v2_fp) if args.verifier_v2_fp else None,
        v3_baseline=Path(args.verifier_v3_baseline) if args.verifier_v3_baseline else None,
        v3_fp=Path(args.verifier_v3_fp) if args.verifier_v3_fp else None,
        threshold=float(args.threshold),
        out_parquet=out,
    )
    print_flag_eval_report(summary, df)
    print(f"Wrote {out}", flush=True)
    return 0


def _cmd_train_crop(args: argparse.Namespace) -> int:
    ckpt = train_crop_from_config(
        Path(args.config),
        run_name=args.run_name,
        tensorboard=not args.no_tensorboard,
    )
    print(f"Checkpoint: {ckpt}")
    return 0


def _cmd_data_build_crop_index(args: argparse.Namespace) -> int:
    chips_index = Path(args.chips_index) if args.chips_index else None
    qa_flags = Path(args.qa_flags) if args.qa_flags else None
    build_crop_index(
        trees_parquet=Path(args.trees),
        chips_index=chips_index,
        out_path=Path(args.out),
        split_meta_path=Path(args.split_meta) if args.split_meta else None,
        qa_flags_path=qa_flags,
        positives_only=bool(args.positives_only),
        grb_negatives_per_chip=int(args.grb_per_chip),
        grb_max_chips=int(args.grb_max_chips),
        limit=int(args.limit) if args.limit else None,
        seed=int(args.seed),
    )
    return 0


def _cmd_data_build_explorer_bundle(args: argparse.Namespace) -> int:
    cli_build_bundle(
        input_path=Path(args.input) if args.input else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        trees_parquet=Path(args.trees) if args.trees else None,
        mark_rgb=bool(args.mark_rgb),
    )
    return 0


def _cmd_serve_explorer(args: argparse.Namespace) -> int:
    from treelidarnet.paths import exports_dir, manifests_dir
    from treelidarnet.qa.crop_service import CropQAServerConfig

    trees_parquet = Path(args.trees) if args.trees else manifests_dir() / "gent_trees.parquet"
    predictions = Path(args.predictions) if args.predictions else None
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else exports_dir() / "inference" / "explorer"
    rescore = bool(args.rescore_on_click)
    cfg = CropQAServerConfig(
        trees_parquet=trees_parquet,
        predictions_export=predictions or (exports_dir() / "inference" / "gent_city_gent_canopy_v2.json.gz"),
        verifier_v2=Path(args.verifier_v2) if args.verifier_v2 else (repo_root() / "runs/gent_crop_verifier_v2/best.pt" if rescore else None),
        verifier_v3=Path(args.verifier_v3) if args.verifier_v3 else (repo_root() / "runs/gent_crop_verifier_v3_rgb/best.pt" if rescore else None),
        radius_m=float(args.radius_m),
        num_points=int(args.num_points),
    )
    serve_explorer(
        host=args.host,
        port=int(args.port),
        crop_cfg=cfg,
        bundle_dir=bundle_dir,
        rescore_on_click=rescore,
    )
    return 0


def _cmd_serve_crop_qa(args: argparse.Namespace) -> int:
    serve_crop_qa(
        host=args.host,
        port=int(args.port),
        trees_parquet=Path(args.trees) if args.trees else None,
        predictions_export=Path(args.predictions) if args.predictions else None,
        verifier_checkpoint=Path(args.verifier) if args.verifier else None,
        verifier_v2=Path(args.verifier_v2) if args.verifier_v2 else None,
        verifier_v3=Path(args.verifier_v3) if args.verifier_v3 else None,
        radius_m=float(args.radius_m),
        num_points=int(args.num_points),
    )
    return 0


def _cmd_filter_export(args: argparse.Namespace) -> int:
    verifier = args.verifier or args.verifier_v3
    if not verifier:
        print("filter-export: provide --verifier or --verifier-v3", file=sys.stderr)
        return 2
    common = dict(
        in_path=Path(args.input),
        out_path=Path(args.output),
        verifier_checkpoint=Path(verifier),
        trees_parquet=Path(args.trees),
        verifier_threshold=float(args.verifier_threshold),
        discovery_m=float(args.discovery_m),
        register_m=float(args.register_m),
        radius_m=float(args.radius_m),
        limit=int(args.limit) if args.limit else None,
        require_chip_rgb=bool(args.require_chip_rgb),
        verifier_checkpoint_v2=Path(args.verifier_v2) if args.verifier_v2 else None,
        verifier_checkpoint_v3=Path(args.verifier_v3) if args.verifier_v3 else None,
        combine_mode=str(args.combine_mode),
        w_v2=float(args.w_v2),
        w_v3=float(args.w_v3),
        infer_batch_size=int(args.infer_batch_size),
        resume=not bool(args.no_resume),
    )
    if args.finalize_live:
        from treelidarnet.filter_export_live import finalize_filter_export_live

        live_dir = Path(args.live_dir) if args.live_dir else None
        pred_path = (
            Path(args.live_pred)
            if args.live_pred
            else (live_dir or exports_dir() / "inference" / "live") / "verified_predictions.json"
        )
        status_path = pred_path.parent / "verified_status.json"
        finalize_filter_export_live(
            in_path=Path(args.input),
            out_path=Path(args.output),
            trees_parquet=Path(args.trees),
            live_pred_path=pred_path,
            status_path=status_path,
            verifier_checkpoint=Path(verifier),
            verifier_threshold=float(args.verifier_threshold),
            bev_min=float(args.bev_min),
            discovery_m=float(args.discovery_m),
            register_m=float(args.register_m),
            verifier_checkpoint_v2=Path(args.verifier_v2) if args.verifier_v2 else None,
            verifier_checkpoint_v3=Path(args.verifier_v3) if args.verifier_v3 else None,
            combine_mode=str(args.combine_mode),
            w_v2=float(args.w_v2),
            w_v3=float(args.w_v3),
        )
    elif args.live:
        from treelidarnet.filter_export_live import filter_export_live

        filter_export_live(**common, bev_min=float(args.bev_min))
    else:
        filter_export(**common)
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
    verifier = Path(args.verifier) if args.verifier else None
    vthresh = float(args.verifier_threshold) if args.verifier else None
    out = infer_chip(
        checkpoint=Path(args.checkpoint),
        chip_id=args.chip_id,
        trees_parquet=Path(args.trees),
        export_path=Path(args.export) if args.export else None,
        model_id=args.model_id,
        score_threshold=float(args.score_threshold),
        verifier_checkpoint=verifier,
        verifier_threshold=vthresh,
        verifier_radius_m=float(args.verifier_radius_m),
    )
    print(f"Export: {out}")
    return 0


def _cmd_qa_checkpoint_center(args: argparse.Namespace) -> int:
    result = compare_center_checkpoints(
        baseline_ckpt=Path(args.baseline),
        candidate_ckpt=Path(args.candidate),
        limit=int(args.limit),
        export_floor=float(args.score_threshold),
        high_thresh=float(args.high_thresh),
    )
    print(result.notes, flush=True)
    print(
        f"PASS={result.passed} "
        f"baseline_rings/chip={result.baseline_rings_per_chip:.2f} "
        f"candidate_rings/chip={result.candidate_rings_per_chip:.2f} "
        f"high_recall={result.high_conf_recall:.3f}",
        flush=True,
    )
    if args.revert_on_fail:
        promote_or_revert(
            result=result,
            baseline_ckpt=Path(args.baseline),
            candidate_ckpt=Path(args.candidate),
        )
    return 0 if result.passed else 1


def _cmd_infer_gent(args: argparse.Namespace) -> int:
    chips_index = Path(args.chips_index) if args.chips_index else None
    if chips_index is not None and not chips_index.is_file():
        chips_index = None
    verifier = Path(args.verifier) if args.verifier else None
    vthresh = float(args.verifier_threshold) if args.verifier else None
    out = infer_gent_city(
        checkpoint=Path(args.checkpoint),
        trees_parquet=Path(args.trees),
        export_path=Path(args.export) if args.export else None,
        model_id=args.model_id,
        batch_size=int(args.batch_size),
        limit=int(args.limit) if args.limit else None,
        chips_index=chips_index,
        live=args.live,
        resume_live=bool(args.resume_live),
        live_flush_every=int(args.live_flush_every),
        progress_every=int(args.progress_every),
        score_threshold=float(args.score_threshold),
        num_workers=int(args.num_workers),
        verifier_checkpoint=verifier,
        verifier_threshold=vthresh,
        verifier_radius_m=float(args.verifier_radius_m),
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
    p_dl.add_argument(
        "--strips",
        action="store_true",
        help="Probe each square and download flight lines the center point missed",
    )
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

    p_ign_patch = sub.add_parser(
        "data-patch-chips-ignore",
        help="Zero the ignore mask on existing chips (run when training is not reading them)",
    )
    p_ign_patch.add_argument("--index", default=None)
    p_ign_patch.add_argument("--limit", default=None)
    p_ign_patch.set_defaults(func=_cmd_data_patch_chips_ignore)

    p_unk_patch = sub.add_parser(
        "data-patch-chips-unknown",
        help="Rewrite unknown mask (default 5 m) on chips that contain a register tree",
    )
    p_unk_patch.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_unk_patch.add_argument("--index", default=None)
    p_unk_patch.add_argument("--band-m", default="5.0")
    p_unk_patch.add_argument("--limit", default=None)
    p_unk_patch.set_defaults(func=_cmd_data_patch_chips_unknown)

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

    p_rgb_patch = sub.add_parser(
        "data-patch-chips-rgb",
        help="Attach DHMV RGB to chip npz (crop verifier; no BEV change)",
    )
    p_rgb_patch.add_argument("--laz-dir", default="data/raw/laz")
    p_rgb_patch.add_argument("--dtm", default=None)
    p_rgb_patch.add_argument("--crops-index", default=None, help="default: crops_index.parquet chip ids")
    p_rgb_patch.add_argument("--chips-index", default=None, help="override chip id list from parquet")
    p_rgb_patch.add_argument("--all-chips", action="store_true", help="scan every npz under data/processed/chips")
    p_rgb_patch.add_argument("--force", action="store_true", help="rewrite rgb even when present")
    p_rgb_patch.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel chip workers (2–4 on a laptop; each reads LAZ)",
    )
    p_rgb_patch.add_argument("--limit", default=None)
    p_rgb_patch.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip first N chip ids (sorted) to resume a full-grid run",
    )
    p_rgb_patch.add_argument(
        "--only-missing-rgb",
        action="store_true",
        help="patch only npz without valid rgb (uses chips_missing_rgb.parquet if present)",
    )
    p_rgb_patch.add_argument(
        "--refresh-missing-manifest",
        action="store_true",
        help="re-scan all chips for missing rgb instead of chips_missing_rgb.parquet",
    )
    p_rgb_patch.add_argument(
        "--register-chips",
        action="store_true",
        help="only chips used by register crop rows in crops_index (train v3 priority)",
    )
    p_rgb_patch.set_defaults(func=_cmd_data_patch_chips_rgb)

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

    p_empty = sub.add_parser(
        "data-rebuild-empty-chips",
        help="Rebuild chips with no points, using every LAZ flight line for the corner",
    )
    p_empty.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_empty.add_argument("--laz-dir", default="data/raw/laz")
    p_empty.add_argument("--index", default=None)
    p_empty.add_argument("--dtm", default=None)
    p_empty.add_argument("--grb-dir", default=None)
    p_empty.add_argument("--road-buffer-m", default="2.0")
    p_empty.add_argument("--limit", default=None)
    p_empty.set_defaults(func=_cmd_data_rebuild_empty_chips)

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
        help="Continue from resume_from in the config (default latest.pt, else best.pt)",
    )
    p_train.set_defaults(func=_cmd_train)

    p_crop_idx = sub.add_parser("data-build-crop-index", help="Build crops_index.parquet for verifier")
    p_crop_idx.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_crop_idx.add_argument(
        "--chips-index",
        default=str(manifests_dir() / "chips_index_full.parquet"),
    )
    p_crop_idx.add_argument("--out", default=str(manifests_dir() / "crops_index.parquet"))
    p_crop_idx.add_argument("--split-meta", default=None)
    p_crop_idx.add_argument(
        "--qa-flags",
        default=None,
        help="Explorer FP flags parquet (default data/manifests/explorer_qa_flags.parquet if exists)",
    )
    p_crop_idx.add_argument("--positives-only", action="store_true")
    p_crop_idx.add_argument("--grb-per-chip", default="2")
    p_crop_idx.add_argument(
        "--grb-max-chips",
        default="15000",
        help="Cap how many chips are scanned for GRB hard-negative samples",
    )
    p_crop_idx.add_argument("--limit", default=None)
    p_crop_idx.add_argument("--seed", default="42")
    p_crop_idx.set_defaults(func=_cmd_data_build_crop_index)

    p_train_crop = sub.add_parser("train-crop", help="Train PointNet++ crop verifier")
    p_train_crop.add_argument("--config", default="configs/gent_crop_verifier_v1.yaml")
    p_train_crop.add_argument("--run-name", default=None)
    p_train_crop.add_argument("--no-tensorboard", action="store_true")
    p_train_crop.set_defaults(func=_cmd_train_crop)

    p_filter = sub.add_parser("filter-export", help="Score and filter a BEV export with the verifier")
    p_filter.add_argument("--input", required=True, help="Input city export .json.gz")
    p_filter.add_argument("--output", required=True, help="Filtered export .json.gz")
    p_filter.add_argument(
        "--verifier",
        default=None,
        help="Crop verifier checkpoint (alias for --verifier-v3)",
    )
    p_filter.add_argument(
        "--verifier-v2",
        default=None,
        help="With --live + --verifier-v3: dual-score map halos (v2 vs v3)",
    )
    p_filter.add_argument("--verifier-v3", default=None, help="RGB verifier checkpoint")
    p_filter.add_argument("--trees", default=str(manifests_dir() / "gent_trees.parquet"))
    p_filter.add_argument(
        "--verifier-threshold",
        default="0.45",
        help="Single-verifier cut, or combo-t for --combine-mode weighted (default 0.45)",
    )
    p_filter.add_argument(
        "--combine-mode",
        default="weighted",
        help="Dual export keep rule: weighted, v3_only, either_or, … (explorer-aligned)",
    )
    p_filter.add_argument("--w-v2", default="0.5", help="Weight for v2 in weighted combine")
    p_filter.add_argument("--w-v3", default="0.5", help="Weight for v3 in weighted combine")
    p_filter.add_argument("--discovery-m", default="5.0")
    p_filter.add_argument("--register-m", default="3.0")
    p_filter.add_argument("--radius-m", default="8.0")
    p_filter.add_argument("--limit", default=None)
    p_filter.add_argument(
        "--live",
        action="store_true",
        help="Stream scored predictions to exports/inference/live/ for crop QA map",
    )
    p_filter.add_argument(
        "--finalize-live",
        action="store_true",
        help="Write --output from existing live verified_predictions.json (skip GPU scoring)",
    )
    p_filter.add_argument(
        "--live-pred",
        default=None,
        help="Path to live verified_predictions.json (default: exports/inference/live/)",
    )
    p_filter.add_argument(
        "--live-dir",
        default=None,
        help="Directory for live JSON when --finalize-live (default: exports/inference/live)",
    )
    p_filter.add_argument(
        "--bev-min",
        default="0.0",
        help="Only score BEV detections with score >= this (e.g. 0.9)",
    )
    p_filter.add_argument(
        "--require-chip-rgb",
        action="store_true",
        help="Only score detections on chips with valid rgb (v3 QA)",
    )
    p_filter.add_argument(
        "--infer-batch-size",
        default="64",
        help="GPU batch size for crop verifier forward (live export)",
    )
    p_filter.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume partial scoring from exports/inference/live/",
    )
    p_filter.set_defaults(func=_cmd_filter_export)

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
    p_infer.add_argument("--verifier", default=None, help="Crop verifier checkpoint")
    p_infer.add_argument(
        "--verifier-threshold",
        default="0.5",
        help="Minimum verifier score when --verifier is set",
    )
    p_infer.add_argument("--verifier-radius-m", default="8.0")
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
    p_city.add_argument(
        "--num-workers",
        default="4",
        help="Processes that build the next grids while the GPU runs",
    )
    p_city.add_argument("--limit", default=None, help="Max chips (smoke test)")
    p_city.add_argument(
        "--live",
        action="store_true",
        help="Write exports/inference/live/ for ghent_live.html polling",
    )
    p_city.add_argument(
        "--resume-live",
        action="store_true",
        help="Continue --live infer from exports/inference/live/status.json (does not reset preds)",
    )
    p_city.add_argument("--live-flush-every", default="150")
    p_city.add_argument("--progress-every", default="50")
    p_city.add_argument(
        "--score-threshold",
        default="0.2",
        help="Minimum model confidence (0–1) to export a peak",
    )
    p_city.add_argument("--verifier", default=None, help="Crop verifier checkpoint path")
    p_city.add_argument(
        "--verifier-threshold",
        default="0.5",
        help="Drop detections below this verifier score when --verifier is set",
    )
    p_city.add_argument("--verifier-radius-m", default="8.0")
    p_city.set_defaults(func=_cmd_infer_gent)

    p_exp_bundle = sub.add_parser(
        "data-build-explorer-bundle",
        help="Columnar explorer bundle from dual-scored export",
    )
    p_exp_bundle.add_argument("--input", default=None, help="verified .json.gz or live verified_predictions.json")
    p_exp_bundle.add_argument("--output-dir", default=None, help="default exports/inference/explorer")
    p_exp_bundle.add_argument("--trees", default=None)
    p_exp_bundle.add_argument(
        "--mark-rgb",
        action="store_true",
        help="Slow: set has_rgb per detection from chip npz",
    )
    p_exp_bundle.set_defaults(func=_cmd_data_build_explorer_bundle)

    p_exp = sub.add_parser(
        "serve-explorer",
        help="Flask end-product QA map + crop API (localhost)",
    )
    p_exp.add_argument("--host", default="127.0.0.1")
    p_exp.add_argument("--port", default="8765")
    p_exp.add_argument("--trees", default=None)
    p_exp.add_argument("--predictions", default=None)
    p_exp.add_argument("--bundle-dir", default=None)
    p_exp.add_argument("--verifier-v2", default=None)
    p_exp.add_argument("--verifier-v3", default=None)
    p_exp.add_argument(
        "--rescore-on-click",
        action="store_true",
        help="Re-run v2/v3 verifier on crop click (default: use export scores only)",
    )
    p_exp.add_argument("--radius-m", default="8.0")
    p_exp.add_argument("--num-points", default="512")
    p_exp.set_defaults(func=_cmd_serve_explorer)

    p_crop_qa = sub.add_parser(
        "serve-crop-qa",
        help="Crop JSON API + Deck.GL / WebGPU viewer (localhost)",
    )
    p_crop_qa.add_argument("--host", default="127.0.0.1")
    p_crop_qa.add_argument("--port", default="8765")
    p_crop_qa.add_argument("--trees", default=None)
    p_crop_qa.add_argument("--predictions", default=None, help="City export .json.gz for pred layer")
    p_crop_qa.add_argument(
        "--verifier",
        default=None,
        help="Legacy: sets v3 checkpoint if --verifier-v3 omitted",
    )
    p_crop_qa.add_argument(
        "--verifier-v2",
        default=None,
        help="6-ch LiDAR verifier (default runs/gent_crop_verifier_v2/best.pt)",
    )
    p_crop_qa.add_argument(
        "--verifier-v3",
        default=None,
        help="9-ch RGB verifier (default runs/gent_crop_verifier_v3_rgb/best.pt)",
    )
    p_crop_qa.add_argument("--radius-m", default="8.0")
    p_crop_qa.add_argument("--num-points", default="512")
    p_crop_qa.set_defaults(func=_cmd_serve_crop_qa)

    p_flag_eval = sub.add_parser(
        "eval-explorer-flags",
        help="Rescore QA false-positive flags: baseline vs FP fine-tuned verifiers",
    )
    p_flag_eval.add_argument(
        "--flags",
        default=str(manifests_dir() / "explorer_qa_flags.parquet"),
    )
    p_flag_eval.add_argument(
        "--verifier-v2-baseline",
        default=str(repo_root() / "runs/gent_crop_verifier_v2/best.pt"),
    )
    p_flag_eval.add_argument(
        "--verifier-v2-fp",
        default=str(repo_root() / "runs/gent_crop_verifier_v2_fp/best.pt"),
    )
    p_flag_eval.add_argument(
        "--verifier-v3-baseline",
        default=str(repo_root() / "runs/gent_crop_verifier_v3_rgb/best.pt"),
    )
    p_flag_eval.add_argument(
        "--verifier-v3-fp",
        default=str(repo_root() / "runs/gent_crop_verifier_v3_rgb_fp/best.pt"),
    )
    p_flag_eval.add_argument("--threshold", default="0.5")
    p_flag_eval.add_argument(
        "--out",
        default=None,
        help="Parquet with per-flag scores (default exports/qa/explorer_fp_rescore.parquet)",
    )
    p_flag_eval.set_defaults(func=_cmd_eval_explorer_flags)

    p_qa = sub.add_parser(
        "qa-checkpoint-center",
        help="Compare baseline vs candidate on central chips (rings/chip, high-conf recall)",
    )
    p_qa.add_argument(
        "--baseline",
        default=str(repo_root() / "runs/gent_canopy_v2/archive/epoch3_hot/best.pt"),
    )
    p_qa.add_argument(
        "--candidate",
        default=str(repo_root() / "runs/gent_canopy_v2/best.pt"),
    )
    p_qa.add_argument("--limit", type=int, default=400)
    p_qa.add_argument("--score-threshold", default="0.2")
    p_qa.add_argument("--high-thresh", default="0.9")
    p_qa.add_argument(
        "--revert-on-fail",
        action="store_true",
        help="Copy baseline back to runs/gent_canopy_v2/best.pt if QA fails",
    )
    p_qa.set_defaults(func=_cmd_qa_checkpoint_center)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
