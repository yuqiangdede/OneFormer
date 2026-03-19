from __future__ import annotations

import unittest

import numpy as np

from src.stats import calc, extract_water_coords


class TestStats(unittest.TestCase):
    def test_calc(self) -> None:
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = calc(mask)

        self.assertEqual(result["water_pixels"], 2)
        self.assertEqual(result["total_pixels"], 4)
        self.assertAlmostEqual(result["ratio"], 0.5)

    def test_extract_water_coords(self) -> None:
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        coords = extract_water_coords(mask)
        self.assertEqual(coords, [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
