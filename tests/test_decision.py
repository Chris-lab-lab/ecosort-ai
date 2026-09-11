import unittest
import math

from ecosort_ai.decision import decide_route


class DecisionTests(unittest.TestCase):
    def test_confident_plastic_routes(self):
        result = decide_route({"plastic": .90, "paper": .04, "metal": .03, "other": .03})
        self.assertEqual(result.route, "plastic")

    def test_general_middle_class_routes(self):
        result = decide_route({"general": .91, "plastic": .04, "metal": .03, "other": .02})
        self.assertEqual(result.route, "general")

    def test_other_never_routes(self):
        result = decide_route({"other": .92, "plastic": .04, "paper": .02, "metal": .02})
        self.assertIsNone(result.route)

    def test_low_margin_does_not_route(self):
        result = decide_route({"plastic": .46, "paper": .42, "metal": .07, "other": .05},
                              confidence_threshold=.4)
        self.assertIsNone(result.route)

    def test_metal_sensor_disagreement_does_not_route(self):
        result = decide_route({"plastic": .9, "paper": .04, "metal": .03, "other": .03},
                              metal_detected=True)
        self.assertIsNone(result.route)

    def test_validity_head_can_reject_a_confident_material(self):
        result = decide_route(
            {"plastic": .94, "general": .04, "metal": .02},
            supported_probability=.18,
            validity_threshold=.6,
        )
        self.assertIsNone(result.route)
        self.assertIn("validity", result.reason)

    def test_distant_feature_embedding_is_rejected(self):
        result = decide_route(
            {"plastic": .94, "general": .04, "metal": .02},
            supported_probability=.95,
            prototype_distance=.31,
            prototype_threshold=.20,
        )
        self.assertIsNone(result.route)
        self.assertIn("feature space", result.reason)

    def test_dual_head_and_feature_checks_accept_known_item(self):
        result = decide_route(
            {"plastic": .94, "general": .04, "metal": .02},
            supported_probability=.95,
            prototype_distance=.08,
            prototype_threshold=.20,
        )
        self.assertEqual(result.route, "plastic")

    def test_nan_score_never_routes(self):
        result = decide_route({"plastic": math.nan, "paper": .4, "metal": .3, "other": .3})
        self.assertIsNone(result.route)
        self.assertIn("invalid", result.reason)

    def test_out_of_range_score_never_routes(self):
        result = decide_route({"plastic": 1.2, "paper": 0.0, "metal": 0.0, "other": 0.0})
        self.assertIsNone(result.route)


if __name__ == "__main__":
    unittest.main()
