from __future__ import annotations

import unittest

import numpy as np

from src.postprocess import extract_water_mask, keep_large


class TestPostprocess(unittest.TestCase):
    def test_extract_water_mask(self) -> None:
        class_map = np.array([[0, 1, 2], [21, 26, 50]], dtype=np.int32)
        water_ids = [21, 26]
        mask = extract_water_mask(class_map, water_ids)

        expected = np.array([[0, 0, 0], [255, 255, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_keep_large(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 255  # area = 4
        mask[4:8, 4:8] = 255  # area = 16

        filtered = keep_large(mask, min_area=5)
        self.assertEqual(int(np.sum(filtered > 0)), 16)


if __name__ == "__main__":
    unittest.main()

