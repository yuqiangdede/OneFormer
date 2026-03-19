from __future__ import annotations

import unittest

import numpy as np

from src.visualize import overlay


class TestVisualize(unittest.TestCase):
    def test_overlay_shape_and_dtype(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :] = [10, 20, 30]
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255

        out = overlay(image, mask, alpha=0.4)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.any(out[mask > 0] != image[mask > 0]))
        self.assertTrue(np.all(out[mask == 0] == image[mask == 0]))


if __name__ == "__main__":
    unittest.main()

