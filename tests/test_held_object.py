from __future__ import annotations

import unittest

from ecosort_ai.held_object import ObjectCandidate, select_held_candidate


class HeldObjectSelectionTests(unittest.TestCase):
    def test_visible_wrist_selects_nearby_object(self) -> None:
        candidates = [
            ObjectCandidate("chair", 0.95, (5, 5, 105, 160)),
            ObjectCandidate("bottle", 0.71, (210, 150, 280, 330)),
        ]

        selected, source = select_held_candidate(
            candidates,
            wrists=[(235, 170)],
            image_shape=(480, 640, 3),
            roi_xyxy=(140, 60, 500, 420),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.label, "bottle")
        self.assertEqual(source, "wrist")

    def test_visible_wrist_does_not_fall_back_to_unrelated_object(self) -> None:
        selected, reason = select_held_candidate(
            [ObjectCandidate("book", 0.90, (260, 180, 360, 300))],
            wrists=[(10, 10)],
            image_shape=(480, 640, 3),
            roi_xyxy=(140, 60, 500, 420),
            max_wrist_distance_ratio=0.05,
        )

        self.assertIsNone(selected)
        self.assertIn("visible wrist", reason)

    def test_center_fallback_ignores_person_and_background_objects(self) -> None:
        selected, source = select_held_candidate(
            [
                ObjectCandidate("person", 0.99, (100, 20, 500, 470)),
                ObjectCandidate("cup", 0.68, (280, 190, 350, 310)),
                ObjectCandidate("chair", 0.97, (510, 100, 630, 450)),
            ],
            wrists=[],
            image_shape=(480, 640, 3),
            roi_xyxy=(140, 60, 500, 420),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.label, "cup")
        self.assertEqual(source, "center")

    def test_center_fallback_can_be_disabled(self) -> None:
        selected, reason = select_held_candidate(
            [ObjectCandidate("cup", 0.80, (280, 190, 350, 310))],
            wrists=[],
            image_shape=(480, 640, 3),
            roi_xyxy=(140, 60, 500, 420),
            allow_center_fallback=False,
        )

        self.assertIsNone(selected)
        self.assertIn("wrist", reason)


if __name__ == "__main__":
    unittest.main()
