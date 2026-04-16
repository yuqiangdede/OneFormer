from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from src.config import INPUT_SIZE, MODEL_LOCAL_DIR
from src.labels import build_oneformer_ade20k_metadata

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


class OneFormerPredictor:
    def __init__(
        self,
        model_dir: str = MODEL_LOCAL_DIR,
        input_size: int = INPUT_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.input_size = input_size
        self.device = torch.device("cpu")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Local model dir not found: {self.model_dir}")

        self._ensure_local_metadata_file()
        self._ensure_local_preprocessor_config()

        try:
            self.processor = OneFormerProcessor.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load local model from: {self.model_dir}"
            ) from e

        self.model.to(self.device)
        self.model.eval()

    def _ensure_local_preprocessor_config(self) -> None:
        config_path = self.model_dir / "preprocessor_config.json"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                config = {}
        else:
            config = {}

        if config.get("repo_path") == str(self.model_dir) and config.get("class_info_file") == "ade20k_panoptic.json":
            return

        config["repo_path"] = str(self.model_dir.resolve())
        config["class_info_file"] = "ade20k_panoptic.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_local_metadata_file(self) -> None:
        metadata_path = self.model_dir / "ade20k_panoptic.json"
        metadata = build_oneformer_ade20k_metadata()

        should_rewrite = True
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                should_rewrite = not self._is_valid_class_info(existing)
            except (json.JSONDecodeError, OSError):
                should_rewrite = True

        if should_rewrite:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _is_valid_class_info(value: object) -> bool:
        if not isinstance(value, dict):
            return False

        for key, item in value.items():
            if key == "thing_ids" or key == "class_names":
                return False
            if not isinstance(item, dict):
                return False
            if "name" not in item or "isthing" not in item:
                return False
        return bool(value)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input image must be HxWx3 RGB")

        target_h, target_w = image_rgb.shape[:2]

        inputs = self.processor(
            images=image_rgb,
            task_inputs=["semantic"],
            size={"height": self.input_size, "width": self.input_size},
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        segmentation = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(target_h, target_w)],
        )[0]

        return segmentation.detach().cpu().numpy().astype(np.int32)
