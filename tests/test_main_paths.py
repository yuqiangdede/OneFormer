from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.image_utils import write_image
from src.main import build_output_paths


class TestMainPaths(unittest.TestCase):
    def test_build_output_paths_with_directory_input(self) -> None:
        input_root = Path("input")
        image_path = Path("input/cam1/a/test.jpg")
        output_root = Path("output")

        original, mask, overlay = build_output_paths(input_root, image_path, output_root)

        self.assertEqual(original, Path("output/cam1/a/test.jpg"))
        self.assertEqual(mask, Path("output/cam1/a/test-mask.png"))
        self.assertEqual(overlay, Path("output/cam1/a/test-overlay.jpg"))

    def test_build_output_paths_with_single_file_input(self) -> None:
        input_root = Path("input/test.jpg")
        image_path = Path("input/test.jpg")
        output_root = Path("output")

        original, mask, overlay = build_output_paths(input_root, image_path, output_root)

        self.assertEqual(original, Path("output/test.jpg"))
        self.assertEqual(mask, Path("output/test-mask.png"))
        self.assertEqual(overlay, Path("output/test-overlay.jpg"))

    def test_write_image_with_non_ascii_path(self) -> None:
        tmp_dir = Path("output/test_unicode")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / "中文-mask.png"
        image = np.zeros((4, 4), dtype=np.uint8)

        self.assertTrue(write_image(target, image))
        self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
