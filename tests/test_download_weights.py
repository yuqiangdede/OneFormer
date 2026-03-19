from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from download_weights import is_model_dir_complete


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
            self.assertTrue(is_model_dir_complete(model_dir))

    def test_complete_with_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "pytorch_model.bin").write_text("fake", encoding="utf-8")
            self.assertTrue(is_model_dir_complete(model_dir))


if __name__ == "__main__":
    unittest.main()

