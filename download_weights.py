from __future__ import annotations

from pathlib import Path

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from src.config import MODEL_LOCAL_DIR, MODEL_NAME

REQUIRED_CONFIG_FILES = ("config.json", "preprocessor_config.json")
REQUIRED_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def is_model_dir_complete(model_dir: Path) -> bool:
    has_configs = all((model_dir / name).exists() for name in REQUIRED_CONFIG_FILES)
    has_weights = any((model_dir / name).exists() for name in REQUIRED_WEIGHT_FILES)
    return has_configs and has_weights


def main() -> None:
    output_dir = Path(MODEL_LOCAL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

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

