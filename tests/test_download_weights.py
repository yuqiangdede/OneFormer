from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

from download_weights import ensure_preprocessor_config, is_model_dir_complete
from src.labels import build_oneformer_ade20k_metadata


class TestDownloadWeights(unittest.TestCase):
    def test_incomplete_without_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.assertFalse(is_model_dir_complete(model_dir))

    def test_complete_with_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
            (model_dir / "ade20k_panoptic.json").write_text("{}", encoding="utf-8")
            self.assertTrue(is_model_dir_complete(model_dir))

    def test_complete_with_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "pytorch_model.bin").write_text("fake", encoding="utf-8")
            (model_dir / "ade20k_panoptic.json").write_text("{}", encoding="utf-8")
            self.assertTrue(is_model_dir_complete(model_dir))

    def test_incomplete_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "pytorch_model.bin").write_text("fake", encoding="utf-8")
            self.assertFalse(is_model_dir_complete(model_dir))

    def test_metadata_helper_shape(self) -> None:
        metadata = build_oneformer_ade20k_metadata()
        self.assertEqual(metadata["0"], "wall")
        self.assertEqual(metadata["149"], "flag")
        self.assertEqual(len(metadata["thing_ids"]), 100)
        self.assertEqual(len(metadata["class_names"]), 150)

    def test_preprocessor_config_is_rewritten_to_local_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            config_path = model_dir / "preprocessor_config.json"
            config_path.write_text(
                json.dumps({"repo_path": "shi-labs/oneformer_demo", "class_info_file": "ade20k_panoptic.json"}),
                encoding="utf-8",
            )

            ensure_preprocessor_config(model_dir)

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["repo_path"], str(model_dir))
            self.assertEqual(config["class_info_file"], "ade20k_panoptic.json")


if __name__ == "__main__":
    unittest.main()

