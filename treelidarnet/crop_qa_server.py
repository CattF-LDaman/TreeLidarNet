"""Local HTTP server: crop JSON API + static viewer."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import torch
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from treelidarnet.data.crops import (
    DEFAULT_CROP_RADIUS_M,
    DEFAULT_MIN_POINTS,
    DEFAULT_NUM_POINTS,
    chip_id_for_point,
    extract_crop_at_global,
)
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.export_mapjson import _tree_radius_m
from treelidarnet.models.crop_verifier import GentCropVerifier, load_crop_verifier
from treelidarnet.paths import chips_dir, exports_dir, manifests_dir, repo_root
from treelidarnet.verifier import score_detection


@dataclass
class CropQAServerConfig:
    trees_parquet: Path
    predictions_export: Path
    verifier_checkpoint: Path | None = None
    verifier_v2: Path | None = None
    verifier_v3: Path | None = None
    radius_m: float = DEFAULT_CROP_RADIUS_M
    num_points: int = DEFAULT_NUM_POINTS
    min_points: int = DEFAULT_MIN_POINTS


def _crop_points_payload(arr: np.ndarray) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in arr:
        pt = {
            "x": float(row[0]),
            "y": float(row[1]),
            "z": float(row[2]),
            "intensity": float(row[3]),
            "return": float(row[4]),
            "class": float(row[5]),
        }
        if row.shape[0] >= 9:
            pt["red"] = float(row[6])
            pt["green"] = float(row[7])
            pt["blue"] = float(row[8])
        out.append(pt)
    return out


def _crop_json(
    *,
    arr: np.ndarray,
    global_x: float,
    global_y: float,
    source: str,
    gis_id: str | None = None,
    score: float | None = None,
    verifier_score: float | None = None,
    radius_m: float,
    num_points: int,
) -> dict[str, Any]:
    return {
        "center": {"x": global_x, "y": global_y},
        "chip_id": chip_id_for_point(global_x, global_y),
        "num_points": int(arr.shape[0]),
        "radius_m": radius_m,
        "source": source,
        "gis_id": gis_id,
        "score": score,
        "verifier_score": verifier_score,
        "verifier_scores": {},
        "points": _crop_points_payload(arr),
    }


def _register_tree_row(row: dict[str, Any]) -> dict[str, Any]:
    st = row.get("stamomtrek")
    radius, has_size = _tree_radius_m(st if st == st else None)
    return {
        "gis_id": row["gis_id"],
        "slug": row.get("gis_slug", ""),
        "sortiment": row.get("sortiment", ""),
        "position": [float(row["lon"]), float(row["lat"])],
        "lambert": {"x": float(row["x"]), "y": float(row["y"])},
        "radiusMeters": radius,
        "hasSize": has_size,
    }


class CropQAService:
    def __init__(self, cfg: CropQAServerConfig) -> None:
        self.cfg = cfg
        self._trees: pd.DataFrame | None = None
        self._register_all: dict[str, Any] | None = None
        self._verifiers: dict[str, GentCropVerifier] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._register_verifier_paths()

    def _register_verifier_paths(self) -> None:
        paths: dict[str, Path | None] = {
            "v2": self.cfg.verifier_v2,
            "v3": self.cfg.verifier_v3,
        }
        if self.cfg.verifier_checkpoint is not None and paths["v3"] is None:
            paths["v3"] = self.cfg.verifier_checkpoint
        self._verifier_paths = paths

    def _verifier_model(self, key: str) -> GentCropVerifier | None:
        path = self._verifier_paths.get(key)
        if path is None:
            return None
        if not path.is_absolute():
            path = repo_root() / path
        if not path.is_file():
            return None
        if key not in self._verifiers:
            self._verifiers[key] = load_crop_verifier(path, self._device)
        return self._verifiers[key]

    def _score_at(
        self, key: str, *, global_x: float, global_y: float, gis_id: str | None
    ) -> float | None:
        model = self._verifier_model(key)
        if model is None:
            return None
        return score_detection(
            model,
            x=global_x,
            y=global_y,
            device=self._device,
            radius_m=self.cfg.radius_m,
            num_points=self.cfg.num_points,
            min_points=self.cfg.min_points,
            seed=hash((global_x, global_y, gis_id, key)) % (2**31),
        )

    def _display_crop_array(
        self, *, global_x: float, global_y: float, gis_id: str | None
    ) -> np.ndarray | None:
        """Prefer 9-channel crop (RGB) for the 3D panel when available."""
        arr = extract_crop_at_global(
            global_x=global_x,
            global_y=global_y,
            radius_m=self.cfg.radius_m,
            num_points=self.cfg.num_points,
            min_points=self.cfg.min_points,
            rng=np.random.default_rng(hash((global_x, global_y, gis_id)) % (2**31)),
            in_channels=9,
        )
        if arr is not None:
            return arr
        return extract_crop_at_global(
            global_x=global_x,
            global_y=global_y,
            radius_m=self.cfg.radius_m,
            num_points=self.cfg.num_points,
            min_points=self.cfg.min_points,
            rng=np.random.default_rng(hash((global_x, global_y, gis_id)) % (2**31)),
            in_channels=6,
        )

    def _load_trees(self) -> pd.DataFrame:
        if self._trees is None:
            if not self.cfg.trees_parquet.is_file():
                self._trees = pd.DataFrame()
            else:
                self._trees = pd.read_parquet(self.cfg.trees_parquet)
        return self._trees

    def health(self) -> dict[str, Any]:
        df = self._load_trees()
        eligible = int(df["eligible"].sum()) if len(df) and "eligible" in df.columns else 0
        loaded = {
            k: self._verifier_model(k) is not None for k in ("v2", "v3")
        }
        paths = {}
        for k, p in self._verifier_paths.items():
            if p is None:
                continue
            paths[k] = str(repo_root() / p if not p.is_absolute() else p)
        return {
            "ok": True,
            "chips_dir": str(chips_dir()),
            "tree_count": eligible,
            "verifier_paths": paths,
            "verifier_loaded": loaded,
        }

    def register_trees_in_bbox(
        self,
        *,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        limit: int,
    ) -> dict[str, Any]:
        df = self._load_trees()
        if df.empty or "eligible" not in df.columns:
            return {"trees": []}
        m = (
            df["eligible"]
            & (df["lon"] >= min_lon)
            & (df["lon"] <= max_lon)
            & (df["lat"] >= min_lat)
            & (df["lat"] <= max_lat)
        )
        sub = df.loc[m].head(limit)
        trees = [_register_tree_row(row) for row in sub.to_dict("records")]
        return {"trees": trees, "count": len(trees)}

    def register_all_trees(self) -> dict[str, Any]:
        if self._register_all is not None:
            return self._register_all
        df = self._load_trees()
        if df.empty or "eligible" not in df.columns:
            self._register_all = {"trees": [], "count": 0}
            return self._register_all
        sub = df.loc[df["eligible"]]
        trees = [_register_tree_row(row) for row in sub.to_dict("records")]
        self._register_all = {"trees": trees, "count": len(trees)}
        return self._register_all

    def predictions_export(self, path: Path | None) -> dict[str, Any]:
        export_path = path or self.cfg.predictions_export
        if not export_path.is_absolute():
            export_path = repo_root() / export_path
        if not export_path.is_file():
            return {"predictions": [], "error": f"not found: {export_path}"}
        with gzip.open(export_path, "rb") as handle:
            payload = orjson.loads(handle.read())
        return {
            "predictions": payload.get("predictions", []),
            "path": str(export_path),
            "model_id": payload.get("model_id"),
            "verifier_checkpoint": payload.get("verifier_checkpoint"),
            "verifier_threshold": payload.get("verifier_threshold"),
        }

    def live_verified_status(self) -> dict[str, Any]:
        status_path = exports_dir() / "inference" / "live" / "verified_status.json"
        if not status_path.is_file():
            return {"phase": "idle", "total": 0, "done": 0}
        return orjson.loads(status_path.read_bytes())

    def live_verified_predictions(self) -> dict[str, Any]:
        pred_path = exports_dir() / "inference" / "live" / "verified_predictions.json"
        if not pred_path.is_file():
            return {"predictions": [], "phase": "idle"}
        return orjson.loads(pred_path.read_bytes())

    def _apply_verifier_scores(
        self,
        payload: dict[str, Any],
        *,
        global_x: float,
        global_y: float,
        gis_id: str | None,
        mode: str,
        row_verifier: float | None = None,
    ) -> dict[str, Any]:
        scores: dict[str, float] = {}
        mode = (mode or "compare").lower()
        if mode == "compare":
            for key in ("v2", "v3"):
                if self._verifier_model(key) is None:
                    continue
                val = self._score_at(key, global_x=global_x, global_y=global_y, gis_id=gis_id)
                if val is not None:
                    scores[key] = val
        elif mode in ("v2", "v3"):
            if self._verifier_model(mode) is not None:
                val = self._score_at(mode, global_x=global_x, global_y=global_y, gis_id=gis_id)
                if val is not None:
                    scores[mode] = val
            elif row_verifier is not None:
                scores[mode] = row_verifier
        payload["verifier_scores"] = scores
        primary = scores.get("v3") if "v3" in scores else scores.get("v2")
        if primary is None and row_verifier is not None and mode != "compare":
            primary = row_verifier
        payload["verifier_score"] = primary
        return payload

    def crop_at_lambert(
        self,
        x: float,
        y: float,
        *,
        source: str,
        gis_id: str | None = None,
        score: float | None = None,
        verifier_score: float | None = None,
        verifier_mode: str = "compare",
    ) -> dict[str, Any] | None:
        arr = self._display_crop_array(global_x=x, global_y=y, gis_id=gis_id)
        if arr is None:
            return None
        payload = _crop_json(
            arr=arr,
            global_x=x,
            global_y=y,
            source=source,
            gis_id=gis_id,
            score=score,
            radius_m=self.cfg.radius_m,
            num_points=self.cfg.num_points,
        )
        return self._apply_verifier_scores(
            payload,
            global_x=x,
            global_y=y,
            gis_id=gis_id,
            mode=verifier_mode,
            row_verifier=verifier_score,
        )

    def crop_for_gis_id(
        self, gis_id: str, *, verifier_mode: str = "compare"
    ) -> dict[str, Any] | None:
        df = self._load_trees()
        row = df.loc[df["gis_id"] == gis_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return self.crop_at_lambert(
            float(r["x"]),
            float(r["y"]),
            source="register",
            gis_id=str(r["gis_id"]),
            verifier_mode=verifier_mode,
        )


def create_crop_qa_app(cfg: CropQAServerConfig) -> Starlette:
    service = CropQAService(cfg)
    viewer_dir = repo_root() / "viewer"

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(service.health())

    async def register_trees(request: Request) -> JSONResponse:
        qp = request.query_params
        try:
            min_lon = float(qp["minLon"])
            min_lat = float(qp["minLat"])
            max_lon = float(qp["maxLon"])
            max_lat = float(qp["maxLat"])
        except KeyError as exc:
            return JSONResponse({"error": f"missing {exc.args[0]}"}, status_code=400)
        limit = int(qp.get("limit", "8000"))
        payload = service.register_trees_in_bbox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            limit=limit,
        )
        return JSONResponse(payload)

    async def register_all(_: Request) -> JSONResponse:
        return JSONResponse(service.register_all_trees())

    async def crop(request: Request) -> Response:
        qp = request.query_params
        vmode = qp.get("verifier_mode", "compare")
        gis_id = qp.get("gis_id")
        if gis_id:
            payload = service.crop_for_gis_id(gis_id, verifier_mode=vmode)
        elif "x" in qp and "y" in qp:
            score = float(qp["score"]) if qp.get("score") else None
            vs = float(qp["verifier_score"]) if qp.get("verifier_score") else None
            payload = service.crop_at_lambert(
                float(qp["x"]),
                float(qp["y"]),
                source=qp.get("source", "prediction"),
                score=score,
                verifier_score=vs,
                verifier_mode=vmode,
            )
        else:
            return JSONResponse({"error": "provide gis_id or x,y"}, status_code=400)
        if payload is None:
            return JSONResponse(
                {"error": "crop unavailable (sparse chip or missing tree)"},
                status_code=404,
            )
        return JSONResponse(payload)

    async def predictions(request: Request) -> JSONResponse:
        rel = request.query_params.get("path")
        path = Path(rel) if rel else None
        return JSONResponse(service.predictions_export(path))

    async def live_status(_: Request) -> JSONResponse:
        return JSONResponse(service.live_verified_status())

    async def live_predictions(_: Request) -> JSONResponse:
        return JSONResponse(service.live_verified_predictions())

    async def crop_from_wgs84(request: Request) -> Response:
        """Convenience: lon/lat → Lambert crop (prediction clicks)."""
        qp = request.query_params
        try:
            lon = float(qp["lon"])
            lat = float(qp["lat"])
        except KeyError as exc:
            return JSONResponse({"error": f"missing {exc.args[0]}"}, status_code=400)
        x, y = wgs84_to_lambert(lon, lat)
        score = float(qp["score"]) if qp.get("score") else None
        vs = float(qp["verifier_score"]) if qp.get("verifier_score") else None
        vmode = qp.get("verifier_mode", "compare")
        payload = service.crop_at_lambert(
            x,
            y,
            source=qp.get("source", "map"),
            score=score,
            verifier_score=vs,
            verifier_mode=vmode,
        )
        if payload is None:
            return JSONResponse({"error": "crop unavailable"}, status_code=404)
        return JSONResponse(payload)

    async def home(_: Request) -> RedirectResponse:
        return RedirectResponse(url="/viewer/crops.html")

    routes = [
        Route("/", home),
        Route("/api/health", health),
        Route("/api/register/trees", register_trees),
        Route("/api/register/all", register_all),
        Route("/api/crop", crop),
        Route("/api/crop/wgs84", crop_from_wgs84),
        Route("/api/predictions", predictions),
        Route("/api/live/verified/status", live_status),
        Route("/api/live/verified/predictions", live_predictions),
        Mount("/viewer", StaticFiles(directory=viewer_dir), name="viewer"),
        Mount("/exports", StaticFiles(directory=exports_dir()), name="exports"),
    ]
    return Starlette(routes=routes)


def serve_crop_qa(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    trees_parquet: Path | None = None,
    predictions_export: Path | None = None,
    verifier_checkpoint: Path | None = None,
    verifier_v2: Path | None = None,
    verifier_v3: Path | None = None,
    radius_m: float = DEFAULT_CROP_RADIUS_M,
    num_points: int = DEFAULT_NUM_POINTS,
) -> None:
    import uvicorn

    trees_parquet = trees_parquet or (manifests_dir() / "gent_trees.parquet")
    predictions_export = predictions_export or (
        exports_dir() / "inference" / "gent_city_gent_canopy_v2.json.gz"
    )
    verifier_v2 = verifier_v2 or (repo_root() / "runs/gent_crop_verifier_v2/best.pt")
    verifier_v3 = verifier_v3 or (repo_root() / "runs/gent_crop_verifier_v3_rgb/best.pt")
    cfg = CropQAServerConfig(
        trees_parquet=trees_parquet,
        predictions_export=predictions_export,
        verifier_checkpoint=verifier_checkpoint,
        verifier_v2=verifier_v2,
        verifier_v3=verifier_v3,
        radius_m=radius_m,
        num_points=num_points,
    )
    app = create_crop_qa_app(cfg)
    print(f"Crop QA: http://{host}:{port}/viewer/crops.html", flush=True)
    print(f"  chips: {chips_dir()}", flush=True)
    print(f"  trees: {trees_parquet}", flush=True)
    print(f"  verifier v2: {verifier_v2}", flush=True)
    print(f"  verifier v3: {verifier_v3}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")