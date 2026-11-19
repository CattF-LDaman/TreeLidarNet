"""Local HTTP server: crop JSON API + static viewer (Starlette)."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from treelidarnet.data.crops import DEFAULT_CROP_RADIUS_M, DEFAULT_NUM_POINTS
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.paths import exports_dir, manifests_dir, repo_root
from treelidarnet.qa.crop_service import CropQAServerConfig, CropQAService

# Re-export for tests and backwards compatibility
__all__ = ["CropQAServerConfig", "CropQAService", "create_crop_qa_app", "serve_crop_qa"]


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

    from treelidarnet.paths import chips_dir

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
    print(f"  explorer: http://{host}:{port}/explorer", flush=True)
    print(f"  chips: {chips_dir()}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")