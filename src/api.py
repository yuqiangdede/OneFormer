from __future__ import annotations

import base64
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from src.config import MIN_AREA, OVERLAY_ALPHA, WATER_LABEL_NAMES
from src.image_utils import bgr_to_rgb
from src.labels import resolve_label_ids
from src.oneformer_infer import OneFormerPredictor
from src.polygons import mask_to_polygons, normalize_polygons
from src.postprocess import extract_water_mask, keep_large, refine_mask
from src.visualize import overlay

app = FastAPI(title="OneFormer Water Seg API")

OUTPUT_API_DIR = Path(os.getenv("OUTPUT_API_DIR", "output/api"))
OUTPUT_RETENTION_HOURS = int(os.getenv("OUTPUT_RETENTION_HOURS", "24"))
OUTPUT_API_DIR.mkdir(parents=True, exist_ok=True)

predictor = OneFormerPredictor()
water_ids, missing_labels = resolve_label_ids(sorted(WATER_LABEL_NAMES))
if not water_ids:
    raise RuntimeError(f"WATER_LABEL_NAMES is invalid, missing={missing_labels}")


class SegmentRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    min_area: int = Field(default=MIN_AREA, ge=0)
    simplify_tolerance: float = Field(default=2.0, ge=0.0)
    alpha: float = Field(default=OVERLAY_ALPHA, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_source(self) -> "SegmentRequest":
        has_url = bool(self.image_url)
        has_b64 = bool(self.image_base64)
        if has_url == has_b64:
            raise ValueError("Provide exactly one of image_url or image_base64")
        return self


def _load_image_from_url(url: str) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {e}") from e

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="image_url content is not a valid image")
    return image


def _load_image_from_base64(image_base64: str) -> np.ndarray:
    payload = image_base64
    if payload.startswith("data:"):
        parts = payload.split(",", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid data URL in image_base64")
        payload = parts[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}") from e

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="image_base64 content is not a valid image")
    return image


def _build_url(file_path: Path, request: Request) -> str:
    relative = file_path.relative_to(OUTPUT_API_DIR).as_posix()
    return str(request.base_url) + f"files/{relative}"


def cleanup_expired_outputs(base_dir: Path, retention_hours: int) -> int:
    now = time.time()
    ttl = max(retention_hours, 0) * 3600
    removed = 0

    for item in base_dir.iterdir():
        try:
            age = now - item.stat().st_mtime
        except FileNotFoundError:
            continue
        if age <= ttl:
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
        except FileNotFoundError:
            continue

    return removed


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/files/{file_path:path}")
def files(file_path: str) -> FileResponse:
    target = OUTPUT_API_DIR / file_path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@app.post("/segment")
def segment(req: SegmentRequest, request: Request) -> dict[str, Any]:
    cleanup_expired_outputs(OUTPUT_API_DIR, OUTPUT_RETENTION_HOURS)

    image_bgr = _load_image_from_url(req.image_url) if req.image_url else _load_image_from_base64(req.image_base64 or "")
    image_rgb = bgr_to_rgb(image_bgr)

    class_map = predictor.predict(image_rgb)
    water_mask = extract_water_mask(class_map, water_ids)
    water_mask = refine_mask(water_mask)
    water_mask = keep_large(water_mask, min_area=req.min_area)
    overlay_img = overlay(image_bgr, water_mask, alpha=req.alpha)

    polygons = mask_to_polygons(water_mask, simplify_tolerance=req.simplify_tolerance)
    height, width = water_mask.shape[:2]
    polygons_normalized = normalize_polygons(polygons, image_width=width, image_height=height)

    task_id = uuid.uuid4().hex
    task_dir = OUTPUT_API_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    mask_path = task_dir / "water_mask.png"
    overlay_path = task_dir / "water_overlay.png"
    polygons_path = task_dir / "water_polygons.json"

    ok1 = cv2.imwrite(str(mask_path), water_mask)
    ok2 = cv2.imwrite(str(overlay_path), overlay_img)
    if not ok1 or not ok2:
        raise HTTPException(status_code=500, detail="Failed to write output images")

    with polygons_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"polygons_xy": polygons_normalized, "polygon_count": len(polygons_normalized)},
            f,
            ensure_ascii=False,
        )

    return {
        "task_id": task_id,
        "mask_url": _build_url(mask_path, request),
        "overlay_url": _build_url(overlay_path, request),
        "polygons_xy": polygons_normalized,
        "polygon_count": len(polygons_normalized),
        "polygons_file": _build_url(polygons_path, request),
    }
