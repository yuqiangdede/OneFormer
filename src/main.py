from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MIN_AREA, OVERLAY_ALPHA, WATER_LABEL_NAMES
from src.image_utils import bgr_to_rgb, collect_images, read_image_bgr, write_image
from src.labels import resolve_label_ids
from src.postprocess import extract_water_mask, keep_large, refine_mask
from src.visualize import overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneFormer 批量水域分割")
    parser.add_argument("--input", default="input", help="输入图片或目录，默认 input")
    parser.add_argument("--output-dir", default="output", help="输出目录，默认 output")
    parser.add_argument("--min-area", type=int, default=MIN_AREA, help=f"连通域最小面积，默认 {MIN_AREA}")
    parser.add_argument(
        "--alpha",
        type=float,
        default=OVERLAY_ALPHA,
        help=f"overlay 透明度，默认 {OVERLAY_ALPHA}",
    )
    return parser.parse_args()


def build_output_paths(input_root: Path, image_path: Path, output_root: Path) -> tuple[Path, Path, Path]:
    relative_path = image_path.relative_to(input_root) if input_root.is_dir() else Path(image_path.name)
    target_original = output_root / relative_path
    target_mask = target_original.with_name(f"{target_original.stem}-mask.png")
    target_overlay = target_original.with_name(f"{target_original.stem}-overlay{target_original.suffix}")
    return target_original, target_mask, target_overlay


def process_one_image(
    predictor,
    water_ids: list[int],
    image_path: Path,
    input_root: Path,
    output_root: Path,
    min_area: int,
    alpha: float,
) -> tuple[Path, Path, Path]:
    image_bgr = read_image_bgr(image_path)
    image_rgb = bgr_to_rgb(image_bgr)

    class_map = predictor.predict(image_rgb)
    water_mask = extract_water_mask(class_map, water_ids)
    water_mask = refine_mask(water_mask)
    water_mask = keep_large(water_mask, min_area=min_area)
    overlay_img = overlay(image_bgr, water_mask, alpha=alpha)

    target_original, target_mask, target_overlay = build_output_paths(input_root, image_path, output_root)
    target_original.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, target_original)
    ok1 = write_image(target_mask, water_mask)
    ok2 = write_image(target_overlay, overlay_img)
    if not ok1 or not ok2:
        raise RuntimeError(f"写入输出图片失败: {image_path}")

    return target_original, target_mask, target_overlay


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.oneformer_infer import OneFormerPredictor
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "缺少依赖，请先安装 requirements.txt（至少包含 transformers）。"
        ) from e

    images = collect_images(input_path)
    predictor = OneFormerPredictor()

    water_ids, missing = resolve_label_ids(sorted(WATER_LABEL_NAMES))
    if not water_ids:
        raise RuntimeError(
            f"未找到任何可用水域标签，缺失: {missing}。请检查 src/labels.py 与 WATER_LABEL_NAMES。"
        )

    for image_path in images:
        target_original, target_mask, target_overlay = process_one_image(
            predictor=predictor,
            water_ids=water_ids,
            image_path=image_path,
            input_root=input_path,
            output_root=output_dir,
            min_area=args.min_area,
            alpha=args.alpha,
        )
        print(f"完成: {target_original}")
        print(f"完成: {target_mask}")
        print(f"完成: {target_overlay}")


if __name__ == "__main__":
    main()
