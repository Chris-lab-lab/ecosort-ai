from __future__ import annotations

import unittest

from ecosort_ai.evaluation import calibrate_validity_threshold, confusion_metrics


class EvaluationTests(unittest.TestCase):
    def test_validity_threshold_separates_supported_from_rejected(self) -> None:
        threshold, balanced_accuracy = calibrate_validity_threshold(
            [0.05, 0.20, 0.82, 0.95],
            [0, 0, 1, 1],
        )

        self.assertGreater(threshold, 0.20)
        self.assertLessEqual(threshold, 0.82)
        self.assertEqual(balanced_accuracy, 1.0)

    def test_validity_threshold_requires_both_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "both"):
            calibrate_validity_threshold([0.8, 0.9], [1, 1])

    def test_confusion_metrics(self) -> None:
        result = confusion_metrics([[8, 2], [1, 9]])

        self.assertAlmostEqual(result["accuracy"], 0.85)
        self.assertAlmostEqual(result["macro_recall"], 0.85)
        self.assertEqual(len(result["per_class"]), 2)


if __name__ == "__main__":
    unittest.main()
