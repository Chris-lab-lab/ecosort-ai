import unittest

import cv2
import numpy as np

from ecosort_ai.mask_refinement import refine_binary_mask, smooth_mask_probability


class MaskRefinementTests(unittest.TestCase):
    def test_temporal_smoothing_blends_probabilities(self) -> None:
        previous = np.zeros((4, 4), dtype=np.float32)
        current = np.ones((4, 4), dtype=np.float32)
        result = smooth_mask_probability(current, previous, current_weight=0.75)
        np.testing.assert_allclose(result, 0.75)

    def test_keeps_center_object_and_removes_islands(self) -> None:
        probability = np.zeros((160, 160), dtype=np.float32)
        probability[45:120, 55:110] = 0.95
        probability[5:30, 5:30] = 0.99
        probability[130:133, 140:143] = 0.99

        result = refine_binary_mask(cv2, probability, threshold=0.5)

        self.assertTrue(result[80, 80])
        self.assertFalse(result[15, 15])
        self.assertFalse(result[131, 141])

    def test_closes_small_crack_but_preserves_background(self) -> None:
        probability = np.zeros((120, 120), dtype=np.float32)
        probability[30:95, 35:90] = 0.9
        probability[60:62, 35:90] = 0.0

        result = refine_binary_mask(cv2, probability, threshold=0.5)

        self.assertTrue(result[60, 60])
        self.assertFalse(result[5, 5])

    def test_empty_mask_stays_empty(self) -> None:
        result = refine_binary_mask(
            cv2, np.zeros((80, 80), dtype=np.float32), threshold=0.5
        )
        self.assertFalse(result.any())


if __name__ == "__main__":
    unittest.main()
