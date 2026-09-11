from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import io

import numpy as np

from ecosort_ai.decision import Decision
from ecosort_ai.live_demo import (
    AnalysisResult, _apply_decision, _validate_args, analyze_frames, average_scores, build_parser,
    run,
)


class FakeClassifier:
    def __init__(self, rows):
        self.rows = iter(rows)

    def predict_rgb(self, _frame):
        return SimpleNamespace(scores=next(self.rows))


class FakeLids:
    def __init__(self) -> None:
        self.calls = []

    def close_all(self) -> None:
        self.calls.append(("close_all",))

    def open_timed(self, route, seconds) -> None:
        self.calls.append(("open_timed", route, seconds))

    def open_temporarily(self, route, seconds) -> None:
        self.calls.append(("open_temporarily", route, seconds))

    def raise_pending_error(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close_all()


class ConsensusTests(unittest.TestCase):
    def test_average_scores_computes_per_class_mean(self) -> None:
        predictions = [
            {"plastic": 0.80, "general": 0.10, "metal": 0.05, "other": 0.05},
            {"plastic": 0.60, "general": 0.20, "metal": 0.10, "other": 0.10},
            {"plastic": 0.70, "general": 0.15, "metal": 0.05, "other": 0.10},
        ]

        result = average_scores(predictions)

        self.assertAlmostEqual(result["plastic"], 0.70)
        self.assertAlmostEqual(result["general"], 0.15)
        self.assertAlmostEqual(result["metal"], 0.20 / 3)
        self.assertAlmostEqual(result["other"], 0.25 / 3)

    def test_label_order_does_not_affect_average(self) -> None:
        result = average_scores(
            [
                {"plastic": 0.8, "other": 0.2},
                {"other": 0.4, "plastic": 0.6},
            ]
        )

        self.assertEqual(list(result), ["plastic", "other"])
        self.assertAlmostEqual(result["plastic"], 0.7)
        self.assertAlmostEqual(result["other"], 0.3)

    def test_empty_consensus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            average_scores([])

    def test_mismatched_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same labels"):
            average_scores([{"plastic": 0.8}, {"plastic": 0.7, "other": 0.3}])

    def test_invalid_scores_are_rejected(self) -> None:
        for bad_score in (-0.01, 1.01, math.nan, math.inf, "0.5", True):
            with self.subTest(score=bad_score):
                with self.assertRaisesRegex(ValueError, "score"):
                    average_scores([{"plastic": bad_score}])

    def test_disagreeing_frames_reject_an_otherwise_confident_average(self) -> None:
        rows = [
            {"plastic": .90, "general": .05, "metal": .03, "other": .02},
            {"plastic": .90, "general": .05, "metal": .03, "other": .02},
            {"plastic": .90, "general": .05, "metal": .03, "other": .02},
            {"plastic": .10, "general": .85, "metal": .03, "other": .02},
            {"plastic": .10, "general": .85, "metal": .03, "other": .02},
        ]
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in rows]

        result = analyze_frames(
            FakeClassifier(rows),
            frames,
            confidence_threshold=.4,
            margin_threshold=.1,
            agreement_threshold=.8,
        )

        self.assertEqual(result.agreement, .6)
        self.assertIsNone(result.decision.route)
        self.assertIn("frames disagree", result.decision.reason)

    def test_stable_frames_accept_a_confident_route(self) -> None:
        rows = [
            {"plastic": .91, "general": .04, "metal": .03, "other": .02}
            for _ in range(5)
        ]
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in rows]

        result = analyze_frames(
            FakeClassifier(rows),
            frames,
            confidence_threshold=.75,
            margin_threshold=.15,
            agreement_threshold=.8,
        )

        self.assertEqual(result.agreement, 1.0)
        self.assertEqual(result.decision.route, "plastic")

    def test_open_set_signals_are_averaged_and_can_reject(self) -> None:
        predictions = iter(
            SimpleNamespace(
                scores={"plastic": .91, "general": .05, "metal": .04},
                supported_probability=supported,
                prototype_distances={"plastic": .12, "general": .20, "metal": .18},
                prototype_thresholds={"plastic": .20, "general": .25, "metal": .22},
            )
            for supported in (.30, .40, .50)
        )
        classifier = SimpleNamespace(predict_rgb=lambda _: next(predictions))
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

        result = analyze_frames(
            classifier,
            frames,
            confidence_threshold=.75,
            margin_threshold=.15,
            validity_threshold=.60,
        )

        self.assertAlmostEqual(result.supported_probability, .40)
        self.assertAlmostEqual(result.prototype_distance, .12)
        self.assertIsNone(result.decision.route)
        self.assertIn("validity head", result.decision.reason)

    def test_metal_sensor_disagreement_rejects_visual_route(self) -> None:
        rows = [{"plastic": .91, "general": .05, "metal": .04} for _ in range(3)]
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in rows]

        result = analyze_frames(
            FakeClassifier(rows),
            frames,
            confidence_threshold=.75,
            margin_threshold=.15,
            metal_detected=True,
        )

        self.assertIsNone(result.decision.route)
        self.assertIn("metal sensor", result.decision.reason)

    def test_live_actuation_returns_deadline_without_sleeping(self) -> None:
        lids = FakeLids()
        result = AnalysisResult(
            scores={"plastic": 0.9, "general": 0.05, "metal": 0.03, "other": 0.02},
            decision=Decision("plastic", 0.9, 0.85, "plastic", "accepted"),
            frame_count=7,
            agreement=1.0,
        )

        with patch("ecosort_ai.live_demo.time.monotonic", return_value=100.0):
            deadline = _apply_decision(lids, result, hold_open=4.0, dry_run=False)

        self.assertEqual(deadline, 104.0)
        self.assertEqual(lids.calls, [("open_timed", "plastic", 4.0)])

    def test_nonfinite_cli_timing_is_rejected(self) -> None:
        for option in ("--hold-open", "--auto-interval"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    parser = build_parser()
                    args = parser.parse_args([
                        "--model", "unused.tflite", "--labels", "unused.txt",
                        f"{option}={value}",
                    ])
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            _validate_args(parser, args)

    def test_manual_analysis_consumes_armed_auto_cycle(self) -> None:
        # A then SPACE before the auto delay must never issue a second opening
        # when time advances beyond that delay.
        keys = iter((ord("a"), 32, ord("q")))
        fake_cv2 = SimpleNamespace(
            imshow=lambda *args: None,
            waitKey=lambda *args: next(keys),
            destroyAllWindows=lambda: None,
        )
        classifier = SimpleNamespace(
            labels=["general", "metal", "other", "plastic"],
            runtime_name="test", delegate_path=None, describe=lambda: "test fixture",
            predict_rgb=lambda _: SimpleNamespace(scores={
                "plastic": 0.91, "general": 0.04, "metal": 0.03, "other": 0.02,
            }),
        )
        lids = FakeLids()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        args = build_parser().parse_args([
            "--model", "unused.tflite", "--labels", "unused.txt",
            "--frames", "1", "--auto-interval", "10", "--dry-run",
        ])
        with (
            patch.dict("sys.modules", {"cv2": fake_cv2}),
            patch("ecosort_ai.live_demo.WasteClassifier", return_value=classifier),
            patch("ecosort_hw.lids.LidController.from_defaults", return_value=lids),
            patch("ecosort_ai.live_demo.OpenCVCamera") as camera,
            patch("ecosort_ai.live_demo._draw_overlay"),
            patch("ecosort_ai.live_demo.time.monotonic", side_effect=(0, 0, 1, 1, 20, 20)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            camera.return_value.__enter__.return_value.read.return_value = (frame, frame)
            run(args)
        openings = [call for call in lids.calls if call[0] == "open_temporarily"]
        self.assertEqual(openings, [("open_temporarily", "plastic", 0)])

    def test_rejected_result_only_closes_lids(self) -> None:
        lids = FakeLids()
        result = AnalysisResult(
            scores={"plastic": 0.4, "general": 0.3, "metal": 0.2, "other": 0.1},
            decision=Decision("plastic", 0.4, 0.1, None, "confidence below threshold"),
            frame_count=7,
            agreement=1.0,
        )

        deadline = _apply_decision(lids, result, hold_open=4.0, dry_run=False)

        self.assertIsNone(deadline)
        self.assertEqual(lids.calls, [("close_all",)])


if __name__ == "__main__":
    unittest.main()
