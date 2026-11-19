"""Flask app: /explorer map + crop API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file

from treelidarnet.data.crops import DEFAULT_CROP_RADIUS_M, DEFAULT_NUM_POINTS
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.explorer.combine import combine_config_from_dict, detection_passes
from treelidarnet.explorer.qa_flags import (
    clear_flag,
    default_flags_path,
    export_flags_snapshot,
    flags_to_json_records,
    load_flags,
    upsert_flag,
)
from treelidarnet.paths import exports_dir, manifests_dir, repo_root
from treelidarnet.qa.crop_service import CropQAServerConfig, CropQAService


def _default_bundle_dir() -> Path:
    return exports_dir() / "inference" / "explorer"


def _bundle_meta(bundle_dir: Path) -> dict:
    path = bundle_dir / "meta.json"
    if not path.is_file():
        return {}
    import orjson

    return orjson.loads(path.read_bytes())


def create_app(
    *,
    crop_cfg: CropQAServerConfig | None = None,
    bundle_dir: Path | None = None,
    rescore_on_click: bool = False,
) -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static"),
        static_url_path="/explorer/static",
    )
    bundle_dir = bundle_dir or _default_bundle_dir()

    if crop_cfg is None:
        trees_parquet = manifests_dir() / "gent_trees.parquet"
        predictions_export = exports_dir() / "inference" / "gent_city_gent_canopy_v2.json.gz"
        crop_cfg = CropQAServerConfig(
            trees_parquet=trees_parquet,
            predictions_export=predictions_export,
            verifier_v2=(
                repo_root() / "runs/gent_crop_verifier_v2/best.pt" if rescore_on_click else None
            ),
            verifier_v3=(
                repo_root() / "runs/gent_crop_verifier_v3_rgb/best.pt" if rescore_on_click else None
            ),
            radius_m=DEFAULT_CROP_RADIUS_M,
            num_points=DEFAULT_NUM_POINTS,
        )
    service = CropQAService(crop_cfg)
    bundle_meta = _bundle_meta(bundle_dir)

    @app.get("/")
    def home():
        return redirect("/explorer")

    @app.get("/explorer")
    def explorer_map():
        return render_template("explorer/map.html", rescore_on_click=rescore_on_click)

    @app.get("/explorer/data/meta.json")
    def explorer_meta():
        path = bundle_dir / "meta.json"
        if not path.is_file():
            return jsonify({"error": "bundle not built; run data-build-explorer-bundle"}), 404
        return Response(path.read_bytes(), mimetype="application/json")

    @app.get("/explorer/data/detections.bin")
    def explorer_detections():
        path = bundle_dir / "detections.bin"
        if not path.is_file():
            return jsonify({"error": "missing detections.bin"}), 404
        resp = send_file(path, mimetype="application/octet-stream", conditional=True)
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

    @app.get("/explorer/data/register.bin")
    def explorer_register():
        path = bundle_dir / "register.bin"
        if not path.is_file():
            return jsonify({"error": "missing register.bin"}), 404
        resp = send_file(path, mimetype="application/octet-stream", conditional=True)
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

    @app.get("/api/health")
    def health():
        payload = service.health()
        payload["explorer"] = {
            "bundle_dir": str(bundle_dir),
            "bundle_meta": bundle_meta,
            "map_scores": "precomputed_bin",
            "crop_scores": "export" if not rescore_on_click else "live_verifier",
        }
        return jsonify(payload)

    @app.get("/api/register/all")
    def register_all():
        return jsonify(service.register_all_trees())

    @app.get("/api/crop")
    def crop():
        vmode = request.args.get("verifier_mode", "compare")
        gis_id = request.args.get("gis_id")
        if gis_id:
            payload = service.crop_for_gis_id(gis_id, verifier_mode=vmode)
        elif "x" in request.args and "y" in request.args:
            score = _float_arg("score")
            vs = _float_arg("verifier_score")
            vv2 = _float_arg("verifier_v2")
            vv3 = _float_arg("verifier_v3")
            use_export = request.args.get("use_export_scores", "").lower() in ("1", "true", "yes")
            if rescore_on_click:
                use_export = False
            elif not rescore_on_click:
                use_export = True
            payload = service.crop_at_lambert(
                float(request.args["x"]),
                float(request.args["y"]),
                source=request.args.get("source", "prediction"),
                score=score,
                verifier_score=vs,
                verifier_v2=vv2,
                verifier_v3=vv3,
                use_export_scores=use_export,
                verifier_mode=vmode,
            )
        else:
            return jsonify({"error": "provide gis_id or x,y"}), 400
        if payload is None:
            return jsonify({"error": "crop unavailable"}), 404
        return jsonify(payload)

    def _float_arg(name: str) -> float | None:
        raw = request.args.get(name)
        return float(raw) if raw not in (None, "") else None

    @app.get("/api/crop/wgs84")
    def crop_wgs84():
        try:
            lon = float(request.args["lon"])
            lat = float(request.args["lat"])
        except KeyError as exc:
            return jsonify({"error": f"missing {exc.args[0]}"}), 400
        x, y = wgs84_to_lambert(lon, lat)
        score = _float_arg("score")
        vs = _float_arg("verifier_score")
        vv2 = _float_arg("verifier_v2")
        vv3 = _float_arg("verifier_v3")
        vmode = request.args.get("verifier_mode", "compare")
        src = request.args.get("source", "map")
        use_export = request.args.get("use_export_scores", "").lower() in ("1", "true", "yes")
        if rescore_on_click:
            use_export = False
        elif not rescore_on_click:
            use_export = True
            if src == "map":
                src = "explorer"
        payload = service.crop_at_lambert(
            x,
            y,
            source=src,
            score=score,
            verifier_score=vs,
            verifier_v2=vv2,
            verifier_v3=vv3,
            use_export_scores=use_export,
            verifier_mode=vmode,
        )
        if payload is None:
            return jsonify({"error": "crop unavailable"}), 404
        return jsonify(payload)

    flags_path = default_flags_path()

    @app.get("/explorer/api/flags")
    def list_flags():
        df = load_flags(flags_path, active_only=True)
        return jsonify({"flags": flags_to_json_records(df), "count": len(df)})

    @app.post("/explorer/api/flags")
    def create_flag():
        body = request.get_json(force=True, silent=True) or {}
        try:
            lon = float(body["lon"])
            lat = float(body["lat"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "lon and lat required"}), 400

        def _opt_float(key: str) -> float | None:
            if key not in body or body[key] is None or body[key] == "":
                return None
            return float(body[key])

        bundle_src = body.get("bundle_source") or bundle_meta.get("source", "")
        row = upsert_flag(
            lon=lon,
            lat=lat,
            bev=_opt_float("bev"),
            v2=_opt_float("v2"),
            v3=_opt_float("v3"),
            dist_register_m=_opt_float("dist_register_m"),
            reason=str(body.get("reason") or "default"),
            note=str(body.get("note") or ""),
            bundle_source=str(bundle_src),
            path=flags_path,
        )
        return jsonify(row)

    @app.delete("/explorer/api/flags/<flag_id>")
    def delete_flag(flag_id: str):
        if clear_flag(flag_id, path=flags_path):
            return jsonify({"ok": True, "flag_id": flag_id})
        return jsonify({"error": "not found"}), 404

    @app.post("/explorer/api/flags/export")
    def export_flags():
        body = request.get_json(force=True, silent=True) or {}
        rel = body.get("path")
        if rel:
            dest = Path(rel)
            if not dest.is_absolute():
                dest = repo_root() / dest
        else:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            dest = exports_dir() / "qa" / f"explorer_fp_{stamp}.parquet"
        n = export_flags_snapshot(dest, source_path=flags_path)
        return jsonify({"path": str(dest), "count": n})

    @app.post("/explorer/api/filter_stats")
    def filter_stats():
        """Server-side counts for combine config (optional sanity check)."""
        body = request.get_json(force=True, silent=True) or {}
        cfg = combine_config_from_dict(body)
        sample = body.get("sample") or []
        kept = 0
        for row in sample:
            if detection_passes(
                bev=float(row.get("bev", 0)),
                v2=float(row.get("v2", 0)),
                v3=float(row.get("v3", 0)),
                dist_register_m=float(row.get("dist_register_m", 0)),
                has_rgb=bool(row.get("has_rgb")),
                cfg=cfg,
            ):
                kept += 1
        return jsonify({"kept": kept, "total": len(sample), "mode": cfg.mode})

    @app.route("/viewer/<path:filename>")
    def legacy_viewer(filename: str):
        return send_file(repo_root() / "viewer" / filename)

    app.config["EXPLORER_BUNDLE_DIR"] = str(bundle_dir)
    app.config["EXPLORER_RESCORE_ON_CLICK"] = rescore_on_click
    return app


def serve_explorer(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    **kwargs,
) -> None:
    app = create_app(**kwargs)
    rescore = app.config.get("EXPLORER_RESCORE_ON_CLICK", False)
    print(f"Explorer: http://{host}:{port}/explorer", flush=True)
    print(f"  bundle: {app.config['EXPLORER_BUNDLE_DIR']}", flush=True)
    crop_scores = "GPU verifier" if rescore else "export only"
    print(f"  scores: map=precomputed bin; crop clicks={crop_scores}", flush=True)
    app.run(host=host, port=port, debug=False, use_reloader=False)