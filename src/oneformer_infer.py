from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from src.config import INPUT_SIZE, MODEL_LOCAL_DIR


class OneFormerPredictor:
    def __init__(
        self,
        model_dir: str = MODEL_LOCAL_DIR,
        input_size: int = INPUT_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.input_size = input_size
        self.device = torch.device("cpu")

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Local model dir not found: {self.model_dir}. "
                "Run: python download_weights.py"
            )

        try:
            self.processor = OneFormerProcessor.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load local model from: {self.model_dir}. "
                "Please rerun: python download_weights.py"
            ) from e

        self.model.to(self.device)
        self.model.eval()

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
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = self.model(**inputs)

        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(target_h, target_w)]
        )[0]
        return segmentation.detach().cpu().numpy().astype(np.int32)
