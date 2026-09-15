from __future__ import annotations

import unittest

import numpy as np

from ecosort_ai.segmentation import (
    composite_masked_object,
    make_background_only,
    select_primary_mask,
)


class SegmentationHelpersTests(unittest.TestCase):
    def test_selects_one_central_object(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 30:70] = True

        decision = select_primary_mask(
            [{"segmentation": mask, "stability_score": 0.95}], mask.shape
        )

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.bbox_xyxy, (30, 25, 70, 75))
        self.assertAlmostEqual(decision.mask_confidence, 0.475)

    def test_rejects_two_distinct_objects(self) -> None:
        first = np.zeros((100, 100), dtype=bool)
        first[25:75, 30:55] = True
        second = np.zeros((100, 100), dtype=bool)
        second[35:70, 70:95] = True

        decision = select_primary_mask(
            [{"segmentation": first}, {"segmentation": second}], first.shape
        )

        self.assertEqual(decision.status, "multiple_objects")
        self.assertEqual(decision.distinct_object_count, 2)

    def test_nested_masks_count_as_one_object(self) -> None:
        outer = np.zeros((100, 100), dtype=bool)
        outer[20:80, 20:80] = True
        inner = np.zeros((100, 100), dtype=bool)
        inner[30:70, 30:70] = True

        decision = select_primary_mask(
            [{"segmentation": outer}, {"segmentation": inner}], outer.shape
        )

        self.assertEqual(decision.status, "accepted")

    def test_can_treat_automatic_masks_as_alternative_proposals(self) -> None:
        first = np.zeros((100, 100), dtype=bool)
        first[25:75, 30:55] = True
        second = np.zeros((100, 100), dtype=bool)
        second[35:70, 70:95] = True

        decision = select_primary_mask(
            [{"segmentation": first}, {"segmentation": second}],
            first.shape,
            reject_multiple_objects=False,
        )

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.distinct_object_count, 1)

    def test_composite_and_background_only(self) -> None:
        rgb = np.full((40, 60, 3), (20, 30, 40), dtype=np.uint8)
        rgb[10:30, 20:40] = (230, 50, 40)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True

        composite = composite_masked_object(
            rgb, mask, output_size=64, rng=np.random.default_rng(4), variant=1
        )
        background = make_background_only(rgb, mask)

        self.assertEqual(composite.shape, (64, 64, 3))
        self.assertTrue(np.all(background[15, 25] == (20, 30, 40)))


if __name__ == "__main__":
    unittest.main()
