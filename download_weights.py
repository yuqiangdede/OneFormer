from __future__ import annotations

import json
from pathlib import Path

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from src.config import MODEL_LOCAL_DIR, MODEL_NAME
from src.labels import build_oneformer_ade20k_metadata

REQUIRED_CONFIG_FILES = ("config.json", "preprocessor_config.json")
REQUIRED_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
REQUIRED_METADATA_FILES = ("ade20k_panoptic.json",)


def is_model_dir_complete(model_dir: Path) -> bool:
    has_configs = all((model_dir / name).exists() for name in REQUIRED_CONFIG_FILES)
    has_weights = any((model_dir / name).exists() for name in REQUIRED_WEIGHT_FILES)
    has_metadata = all((model_dir / name).exists() for name in REQUIRED_METADATA_FILES)
    return has_configs and has_weights and has_metadata


def ensure_metadata_file(output_dir: Path) -> Path:
    metadata_path = output_dir / "ade20k_panoptic.json"
    metadata = build_oneformer_ade20k_metadata()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    output_dir = Path(MODEL_LOCAL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_metadata_file(output_dir)

    if is_model_dir_complete(output_dir):
        print(f"[SKIP] Model already exists and is complete: {output_dir}")
        return

    print(f"[DOWNLOAD] processor: {MODEL_NAME}")
    processor = OneFormerProcessor.from_pretrained(MODEL_NAME)
    processor.save_pretrained(str(output_dir))

    print(f"[DOWNLOAD] model: {MODEL_NAME}")
    model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.save_pretrained(str(output_dir))

    if not is_model_dir_complete(output_dir):
        raise RuntimeError(f"Download finished but model dir is incomplete: {output_dir}")

    print(f"[DONE] Saved to: {output_dir}")


if __name__ == "__main__":
    main()

