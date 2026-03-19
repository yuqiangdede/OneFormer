from __future__ import annotations

import numpy as np


def overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr 必须是 HxWx3")
    if mask.ndim != 2:
        raise ValueError("mask 必须是 HxW")

    color = np.zeros_like(image_bgr, dtype=np.uint8)
    color[:, :, 0] = 255  # BGR: 蓝色

    out = image_bgr.astype(np.float32).copy()
    mask_pos = mask > 0
    out[mask_pos] = out[mask_pos] * (1.0 - alpha) + color[mask_pos].astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)

