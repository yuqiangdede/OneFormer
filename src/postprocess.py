from __future__ import annotations

import cv2
import numpy as np


def extract_water_mask(class_map: np.ndarray, water_ids: list[int]) -> np.ndarray:
    mask = np.isin(class_map, water_ids)
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def keep_large(mask: np.ndarray, min_area: int = 300) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > min_area:
            new_mask[labels == idx] = 255
    return new_mask

