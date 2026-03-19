from __future__ import annotations

import unittest

import numpy as np

from src.polygons import mask_to_polygons, normalize_polygons


class TestPolygons(unittest.TestCase):
    def test_mask_to_polygons(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 3:7] = 255
        polygons = mask_to_polygons(mask, simplify_tolerance=0.0)
        self.assertGreaterEqual(len(polygons), 1)
        self.assertGreaterEqual(len(polygons[0]), 4)

    def test_normalize_polygons(self) -> None:
        polygons = [[[10, 20], [50, 20], [50, 60], [10, 60]]]
        normalized = normalize_polygons(polygons, image_width=100, image_height=200)
        self.assertEqual(normalized[0][0], [0.1, 0.1])
        self.assertEqual(normalized[0][2], [0.5, 0.3])


if __name__ == "__main__":
    unittest.main()
