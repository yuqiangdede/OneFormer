from __future__ import annotations

import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, simplify_tolerance: float = 2.0) -> list[list[list[int]]]:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[int]]] = []

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        epsilon = max(float(simplify_tolerance), 0.0)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] < 3:
            continue
        points = [[int(p[0][0]), int(p[0][1])] for p in approx]
        polygons.append(points)

    return polygons


def normalize_polygons(
    polygons: list[list[list[int]]],
    image_width: int,
    image_height: int,
) -> list[list[list[float]]]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    out: list[list[list[float]]] = []
    for polygon in polygons:
        poly: list[list[float]] = []
        for x, y in polygon:
            nx = float(x) / float(image_width)
            ny = float(y) / float(image_height)
            poly.append([nx, ny])
        out.append(poly)
    return out
