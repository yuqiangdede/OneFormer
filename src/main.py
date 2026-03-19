from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MIN_AREA, OVERLAY_ALPHA, WATER_LABEL_NAMES
from src.image_utils import bgr_to_rgb, read_image_bgr
from src.labels import resolve_label_ids
from src.postprocess import extract_water_mask, keep_large, refine_mask
from src.stats import calc, extract_water_coords
from src.visualize import overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneFormer 单图水域分割")
    parser.add_argument("--image", required=True, help="输入图片路径（jpg/png）")
    parser.add_argument("--output-dir", default="output", help="输出目录，默认 output")
    parser.add_argument("--min-area", type=int, default=MIN_AREA, help=f"连通域最小面积，默认 {MIN_AREA}")
    parser.add_argument(
        "--alpha",
        type=float,
        default=OVERLAY_ALPHA,
        help=f"overlay 透明度，默认 {OVERLAY_ALPHA}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.oneformer_infer import OneFormerPredictor
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "缺少依赖，请先安装 requirements.txt（至少包含 transformers）。"
        ) from e

    image_bgr = read_image_bgr(args.image)
    image_rgb = bgr_to_rgb(image_bgr)

    predictor = OneFormerPredictor()
    class_map = predictor.predict(image_rgb)

    water_ids, missing = resolve_label_ids(sorted(WATER_LABEL_NAMES))
    if not water_ids:
        raise RuntimeError(
            f"未找到任何可用水域标签，缺失: {missing}。请检查 src/labels.py 与 WATER_LABEL_NAMES。"
        )

    water_mask = extract_water_mask(class_map, water_ids)
    water_mask = refine_mask(water_mask)
    water_mask = keep_large(water_mask, min_area=args.min_area)

    overlay_img = overlay(image_bgr, water_mask, alpha=args.alpha)
    stats = calc(water_mask)

    mask_path = output_dir / "water_mask.png"
    overlay_path = output_dir / "water_overlay.png"
    stats_path = output_dir / "stats.json"
    coords_path = output_dir / "water_coords.json"

    ok1 = cv2.imwrite(str(mask_path), water_mask)
    ok2 = cv2.imwrite(str(overlay_path), overlay_img)
    if not ok1 or not ok2:
        raise RuntimeError("写入输出图片失败，请检查输出目录权限。")

    water_coords = extract_water_coords(water_mask)
    with coords_path.open("w", encoding="utf-8") as f:
        json.dump({"coords_xy": water_coords}, f, ensure_ascii=False)

    stats["coords_file"] = str(coords_path)
    stats["coords_count"] = len(water_coords)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"完成: {mask_path}")
    print(f"完成: {overlay_path}")
    print(f"完成: {coords_path}")
    print(f"完成: {stats_path}")
    print(f"water_ids={water_ids}, missing={missing}")
    print(f"stats={stats}")


if __name__ == "__main__":
    main()
