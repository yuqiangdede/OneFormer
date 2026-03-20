from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def collect_images(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"不支持的图片类型: {input_path}")
        return [input_path]

    images = sorted(
        path for path in input_path.rglob("*") if path.is_file() and is_image_file(path)
    )
    if not images:
        raise ValueError(f"目录内未找到图片: {input_path}")
    return images


def write_image(image_path: str | Path, image: np.ndarray) -> bool:
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    ext = suffix if suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(image_path))
    return True


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
