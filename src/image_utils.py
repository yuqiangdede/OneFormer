from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_numpy(
    image_rgb: np.ndarray,
    input_size: int,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """纯 numpy 预处理流程（用于调试/对齐，不参与 OneFormer 主推理）。"""
    resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean_arr) / std_arr
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x

