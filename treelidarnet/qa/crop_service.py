"""Crop QA service: register, crops, dual verifier scoring."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import torch

from treelidarnet.data.crops import (
    DEFAULT_CROP_RADIUS_M,
    DEFAULT_MIN_POINTS,
    DEFAULT_NUM_POINTS,
    chip_id_for_point,
    extract_crop_at_global,
)
from treelidarnet.export_mapjson import _tree_radius_m
from treelidarnet.models.crop_verifier import GentCropVerifier, load_crop_verifier
from treelidarnet.paths import chips_dir, exports_dir, repo_root
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


def crop_points_payload(arr: np.ndarray) -> list[dict[str, float]]:
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


def crop_json(
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
        "points": crop_points_payload(arr),
    }


def register_tree_row(row: dict[str, Any]) -> dict[str, Any]:
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
        loaded = {k: self._verifier_model(k) is not None for k in ("v2", "v3")}
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
        trees = [register_tree_row(row) for row in sub.to_dict("records")]
        return {"trees": trees, "count": len(trees)}

    def register_all_trees(self) -> dict[str, Any]:
        if self._register_all is not None:
            return self._register_all
        df = self._load_trees()
        if df.empty or "eligible" not in df.columns:
            self._register_all = {"trees": [], "count": 0}
            return self._register_all
        sub = df.loc[df["eligible"]]
        trees = [register_tree_row(row) for row in sub.to_dict("records")]
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

    def _attach_export_scores(
        self,
        payload: dict[str, Any],
        *,
        verifier_v2: float | None,
        verifier_v3: float | None,
        verifier_score: float | None,
    ) -> dict[str, Any]:
        scores: dict[str, float] = {}
        if verifier_v2 is not None:
            scores["v2"] = verifier_v2
        if verifier_v3 is not None:
            scores["v3"] = verifier_v3
        payload["verifier_scores"] = scores
        primary = scores.get("v3")
        if primary is None:
            primary = scores.get("v2")
        if primary is None and verifier_score is not None:
            primary = verifier_score
        payload["verifier_score"] = primary
        payload["scores_from_export"] = True
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
        verifier_v2: float | None = None,
        verifier_v3: float | None = None,
        use_export_scores: bool = False,
        verifier_mode: str = "compare",
    ) -> dict[str, Any] | None:
        arr = self._display_crop_array(global_x=x, global_y=y, gis_id=gis_id)
        if arr is None:
            return None
        payload = crop_json(
            arr=arr,
            global_x=x,
            global_y=y,
            source=source,
            gis_id=gis_id,
            score=score,
            verifier_score=verifier_score,
            radius_m=self.cfg.radius_m,
            num_points=self.cfg.num_points,
        )
        if use_export_scores:
            return self._attach_export_scores(
                payload,
                verifier_v2=verifier_v2,
                verifier_v3=verifier_v3,
                verifier_score=verifier_score,
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