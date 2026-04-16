from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.oneformer_infer import OneFormerPredictor


class TestOneFormerInfer(unittest.TestCase):
    def test_missing_preprocessor_config_is_created_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            predictor = object.__new__(OneFormerPredictor)
            predictor.model_dir = model_dir.resolve()

            predictor._ensure_local_metadata_file()
            predictor._ensure_local_preprocessor_config()

            metadata_path = model_dir / "ade20k_panoptic.json"
            self.assertTrue(metadata_path.exists())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["0"]["name"], "wall")
            self.assertFalse(metadata["0"]["isthing"])

            config_path = model_dir / "preprocessor_config.json"
            self.assertTrue(config_path.exists())

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["repo_path"], str(model_dir.resolve()))
            self.assertEqual(config["class_info_file"], "ade20k_panoptic.json")

    def test_warmup_runs_single_dummy_inference(self) -> None:
        predictor = object.__new__(OneFormerPredictor)
        predictor.input_size = 64
        predictor.predict = Mock(return_value=np.zeros((64, 64), dtype=np.int32))

        predictor.warmup()

        predictor.predict.assert_called_once()
        warmup_image = predictor.predict.call_args.args[0]
        self.assertEqual(warmup_image.shape, (64, 64, 3))
        self.assertEqual(warmup_image.dtype, np.uint8)
        self.assertTrue(np.all(warmup_image == 0))


if __name__ == "__main__":
    unittest.main()
