from __future__ import annotations

import unittest

import numpy as np

from dataset_studio import centered_square
from ecosort_ai.app import centered_roi


class DatasetStudioTests(unittest.TestCase):
    def test_centered_square_from_wide_frame(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        roi, bounds = centered_square(frame)

        self.assertEqual(roi.shape, (72, 72, 3))
        self.assertEqual(bounds, (64, 14, 136, 86))

    def test_runtime_uses_same_centered_roi_geometry(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        roi, placement = centered_roi(frame)

        self.assertEqual(roi.shape, (72, 72, 3))
        self.assertEqual(placement, (64, 14, 72))


if __name__ == "__main__":
    unittest.main()
