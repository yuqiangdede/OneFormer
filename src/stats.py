from __future__ import annotations

import numpy as np


def calc(mask: np.ndarray) -> dict:
    total = int(mask.size)
    water = int(np.sum(mask > 0))
    ratio = float(water / total) if total else 0.0
    return {
        "water_pixels": water,
        "total_pixels": total,
        "ratio": ratio,
    }


def extract_water_coords(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.where(mask > 0)
    return [[int(x), int(y)] for y, x in zip(ys, xs)]
