from __future__ import annotations

from pathlib import Path

from src.labels import WATER_RELATED_LABEL_NAMES

MODEL_NAME = "shi-labs/oneformer_ade20k_swin_large"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_LOCAL_DIR = PROJECT_ROOT / "models" / "oneformer_ade20k_swin_large"

INPUT_SIZE = 512

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

WATER_LABEL_NAMES = set(WATER_RELATED_LABEL_NAMES)

OVERLAY_ALPHA = 0.4
MIN_AREA = 300
