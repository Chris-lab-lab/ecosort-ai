"""Interactive OpenCV demo for the EcoSort classifier and three-lid bin.

The demo starts in dry-run mode.  Hardware movement is enabled only with the
explicit ``--live`` flag and an explicit I2C bus number.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import math
import numbers
from pathlib import Path
import time
from typing import Any

import numpy as np

from .camera import OpenCVCamera
from .decision import Decision, decide_route
from .model import WasteClassifier


CORRECTION_KEYS = {
    ord("p"): "plastic",
    ord("g"): "general",
    ord("m"): "metal",
    ord("o"): "other",
}
SAFE_LID_ROUTES = frozenset({"plastic", "paper", "general", "metal"})


def average_scores(score_maps: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Return the per-class arithmetic mean for a sequence of predictions.

    Every prediction must be a non-empty mapping with exactly the same labels.
    Scores must be real, finite probabilities in the inclusive range 0..1.
    The function is deliberately independent of the model, camera, and UI so
    the temporal-consensus safety rule can be tested without hardware.
    """

    predictions = tuple(score_maps)
    if not predictions:
        raise ValueError("at least one prediction is required")
    if not isinstance(predictions[0], Mapping) or not predictions[0]:
        raise ValueError("each prediction must be a non-empty score mapping")

    labels = tuple(predictions[0])
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("prediction labels must be non-empty strings")
    expected_labels = set(labels)
    rows: list[list[float]] = []

    for index, prediction in enumerate(predictions, start=1):
        if not isinstance(prediction, Mapping) or not prediction:
            raise ValueError(f"prediction {index} must be a non-empty score mapping")
        if set(prediction) != expected_labels:
            raise ValueError("all predictions must contain exactly the same labels")

        row: list[float] = []
        for label in labels:
            raw_score = prediction[label]
            if isinstance(raw_score, bool) or not isinstance(raw_score, numbers.Real):
                raise ValueError(f"score for {label!r} in prediction {index} is not numeric")
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"score for {label!r} in prediction {index} must be finite and between 0 and 1"
                )
            row.append(score)
        rows.append(row)

    means = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    return {label: float(means[offset]) for offset, label in enumerate(labels)}


@dataclass(frozen=True)
class AnalysisResult:
    scores: dict[str, float]
    decision: Decision
    frame_count: int
    agreement: float
    supported_probability: float | None = None
    prototype_distance: float | None = None
    prototype_threshold: float | None = None
    metal_detected: bool | None = None


def analyze_frames(
    classifier: WasteClassifier,
    rgb_frames: Iterable[np.ndarray],
    *,
    confidence_threshold: float,
    margin_threshold: float,
    agreement_threshold: float = 0.8,
    validity_threshold: float = 0.5,
    metal_detected: bool | None = None,
) -> AnalysisResult:
    """Infer several ROI frames and apply the normal conservative router."""

    frames = tuple(rgb_frames)
    frame_predictions = [classifier.predict_rgb(frame) for frame in frames]
    score_rows = [prediction.scores for prediction in frame_predictions]
    scores = average_scores(score_rows)
    winner = max(scores, key=scores.get)

    validity_rows = [getattr(prediction, "supported_probability", None) for prediction in frame_predictions]
    supported_probability = (
        float(np.mean(validity_rows)) if all(value is not None for value in validity_rows) else None
    )
    distance_rows = [getattr(prediction, "prototype_distances", None) for prediction in frame_predictions]
    prototype_distance = (
        float(np.mean([row[winner] for row in distance_rows]))
        if all(row is not None and winner in row for row in distance_rows)
        else None
    )
    threshold_rows = [getattr(prediction, "prototype_thresholds", None) for prediction in frame_predictions]
    prototype_threshold = (
        float(threshold_rows[0][winner])
        if threshold_rows and threshold_rows[0] is not None and winner in threshold_rows[0]
        else None
    )
    decision = decide_route(
        scores,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        supported_probability=supported_probability,
        validity_threshold=validity_threshold,
        prototype_distance=prototype_distance,
        prototype_threshold=prototype_threshold,
        metal_detected=metal_detected,
    )
    agreement = sum(max(row, key=row.get) == winner for row in score_rows) / len(score_rows)
    if decision.accepted and agreement < agreement_threshold:
        decision = Decision(
            label=decision.label,
            confidence=decision.confidence,
            margin=decision.margin,
            route=None,
            reason="frames disagree; hold the item steady",
        )
    return AnalysisResult(
        scores=scores,
        decision=decision,
        frame_count=len(frames),
        agreement=agreement,
        supported_probability=supported_probability,
        prototype_distance=prototype_distance,
        prototype_threshold=prototype_threshold,
        metal_detected=metal_detected,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advanced EcoSort OpenCV live demo with temporal consensus"
    )
    parser.add_argument("--model", required=True, help="TFLite or Vela-compiled model")
    parser.add_argument("--labels", required=True, help="One label per model output line")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="optional open_set.json path (auto-detected beside the model)",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--delegate", default="auto", help="auto, none, or delegate .so path")
    parser.add_argument("--threads", type=int, default=2, help="TFLite CPU thread count")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument(
        "--validity-threshold",
        type=float,
        help="override the calibrated supported-item threshold stored in open_set.json",
    )
    parser.add_argument(
        "--agreement",
        type=float,
        default=0.8,
        help="minimum fraction of frames voting for the winning class",
    )
    parser.add_argument(
        "--average-frames",
        "--frames",
        dest="average_frames",
        type=int,
        default=5,
        metavar="N",
        help="average the latest N ROI predictions (default: 5)",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=0.72,
        help="square ROI size as a fraction of the shortest frame side",
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=2.0,
        help="seconds the item must remain presented after auto is armed",
    )
    parser.add_argument("--hold-open", type=float, default=4.0)
    parser.add_argument(
        "--corrections-dir",
        type=Path,
        default=Path("corrections"),
        help="root folder for ROI correction examples",
    )
    parser.add_argument("--i2c-bus", type=int, help="required when --live is used")
    parser.add_argument("--i2c-address", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument(
        "--hardware-config",
        type=Path,
        help="JSON file with per-lid PCA9685 channels and calibrated pulses",
    )
    parser.add_argument(
        "--metal-sensor-path",
        type=Path,
        help="read an already-configured digital metal sensor value file on Linux",
    )
    parser.add_argument(
        "--metal-sensor-active-low",
        action="store_true",
        help="treat a low digital sensor value as metal detected",
    )

    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="show decisions without moving lids (default)",
    )
    hardware.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="enable physical lids; requires --i2c-bus",
    )
    parser.set_defaults(dry_run=True)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if not 0.0 <= args.margin <= 1.0:
        parser.error("--margin must be between 0 and 1")
    if args.validity_threshold is not None and not 0.0 <= args.validity_threshold <= 1.0:
        parser.error("--validity-threshold must be between 0 and 1")
    if not 0.0 <= args.agreement <= 1.0:
        parser.error("--agreement must be between 0 and 1")
    if args.average_frames < 1:
        parser.error("--average-frames must be at least 1")
    if not 0.1 <= args.roi_scale <= 1.0:
        parser.error("--roi-scale must be between 0.1 and 1.0")
    if not math.isfinite(args.auto_interval) or args.auto_interval < 0:
        parser.error("--auto-interval must be finite and non-negative")
    if not math.isfinite(args.hold_open) or args.hold_open < 0:
        parser.error("--hold-open must be finite and non-negative")
    if args.camera_width < 1 or args.camera_height < 1:
        parser.error("camera dimensions must be positive")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.i2c_bus is not None and args.i2c_bus < 0:
        parser.error("--i2c-bus must be zero or greater")
    if not args.dry_run and args.i2c_bus is None:
        parser.error("--live requires an explicit --i2c-bus")


def _centered_square(frame: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    side = max(1, int(min(width, height) * scale))
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def _backend_name(classifier: WasteClassifier) -> str:
    accelerator = (
        Path(classifier.delegate_path).name if classifier.delegate_path else "CPU"
    )
    return f"{classifier.runtime_name} / {accelerator}"


def _put_text(
    cv2: Any,
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = (240, 240, 240),
    scale: float = 0.58,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_overlay(
    cv2: Any,
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    backend: str,
    dry_run: bool,
    auto_mode: bool,
    buffered_frames: int,
    required_frames: int,
    result: AnalysisResult | None,
    status: str,
) -> None:
    height, width = frame.shape[:2]
    left, top, right, bottom = roi
    mode_color = (0, 210, 255) if auto_mode else (100, 230, 100)
    cv2.rectangle(frame, (left, top), (right, bottom), mode_color, 2)
    _put_text(
        cv2,
        frame,
        "CENTER ONE ITEM HERE",
        (left + 8, max(24, top - 10)),
        color=mode_color,
        scale=0.55,
        thickness=2,
    )

    panel_width = min(width, 570)
    panel_height = 294 if result else 126
    panel = frame[0 : min(panel_height, height), 0:panel_width]
    shade = np.zeros_like(panel)
    cv2.addWeighted(shade, 0.58, panel, 0.42, 0, panel)

    operating_mode = "AUTO" if auto_mode else "MANUAL"
    hardware_mode = "DRY RUN" if dry_run else "LIVE HARDWARE"
    _put_text(
        cv2,
        frame,
        f"EcoSort | {operating_mode} | {hardware_mode}",
        (14, 25),
        color=mode_color,
        scale=0.64,
        thickness=2,
    )
    _put_text(cv2, frame, f"Backend: {backend}", (14, 49))
    _put_text(
        cv2,
        frame,
        f"Consensus frames: {buffered_frames}/{required_frames}",
        (14, 72),
    )

    if result is not None:
        ranked = sorted(result.scores.items(), key=lambda item: item[1], reverse=True)
        bar_left = 108
        bar_width = max(70, panel_width - 185)
        bar_colors = {
            "plastic": (115, 205, 105),
            "general": (220, 150, 70),
            "paper": (220, 150, 70),
            "metal": (125, 145, 235),
            "other": (145, 145, 145),
        }
        for row, (label, score) in enumerate(ranked[:4]):
            y = 91 + row * 25
            _put_text(cv2, frame, label.upper(), (14, y + 12), scale=0.49)
            cv2.rectangle(
                frame,
                (bar_left, y),
                (bar_left + bar_width, y + 14),
                (65, 65, 65),
                cv2.FILLED,
            )
            cv2.rectangle(
                frame,
                (bar_left, y),
                (bar_left + int(bar_width * score), y + 14),
                bar_colors.get(label, (120, 200, 160)),
                cv2.FILLED,
            )
            _put_text(
                cv2,
                frame,
                f"{score:.0%}",
                (bar_left + bar_width + 8, y + 12),
                scale=0.46,
            )
        _put_text(
            cv2,
            frame,
            f"Confidence: {result.decision.confidence:.1%}  margin: {result.decision.margin:.1%}  frames: {result.agreement:.0%}",
            (14, 208),
            scale=0.50,
        )
        open_set_bits = []
        if result.supported_probability is not None:
            open_set_bits.append(f"supported: {result.supported_probability:.1%}")
        if result.prototype_distance is not None and result.prototype_threshold is not None:
            open_set_bits.append(
                f"feature distance: {result.prototype_distance:.3f}/{result.prototype_threshold:.3f}"
            )
        if result.metal_detected is not None:
            open_set_bits.append(f"metal sensor: {'YES' if result.metal_detected else 'no'}")
        if open_set_bits:
            _put_text(cv2, frame, "  ".join(open_set_bits), (14, 231), scale=0.47)
        if result.decision.accepted and result.decision.route in SAFE_LID_ROUTES:
            decision_text = f"Decision: OPEN {result.decision.route.upper()}"
            decision_color = (70, 235, 70)
        else:
            decision_text = f"Decision: REJECT - {result.decision.reason}"
            decision_color = (70, 160, 255)
        _put_text(
            cv2,
            frame,
            decision_text,
            (14, 257),
            color=decision_color,
            scale=0.57,
            thickness=2,
        )

    if status:
        _put_text(cv2, frame, status, (14, panel_height - 12), color=(255, 220, 100))

    help_text = "SPACE analyze | A arm auto | P/G/M/O save correction | Q/ESC quit"
    help_y = max(22, height - 16)
    (text_width, _), _ = cv2.getTextSize(
        help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    )
    help_left = max(0, min(10, width - text_width - 12))
    cv2.rectangle(
        frame,
        (0, max(0, help_y - 20)),
        (width, height),
        (18, 18, 18),
        cv2.FILLED,
    )
    _put_text(cv2, frame, help_text, (help_left, help_y), scale=0.52)


def _save_correction(
    cv2: Any,
    roi_bgr: np.ndarray,
    root: Path,
    label: str,
) -> Path:
    destination = root / label
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = destination / f"correction_{stamp}.jpg"
    if not cv2.imwrite(str(path), roi_bgr):
        raise OSError(f"OpenCV could not write {path}")
    return path


def _print_result(result: AnalysisResult, *, dry_run: bool) -> None:
    ranked = sorted(result.scores.items(), key=lambda item: item[1], reverse=True)
    summary = ", ".join(f"{label}={score:.1%}" for label, score in ranked)
    print(
        f"Consensus ({result.frame_count} frames, {result.agreement:.0%} winner agreement): "
        f"{summary}"
    )
    diagnostics = []
    if result.supported_probability is not None:
        diagnostics.append(f"supported={result.supported_probability:.1%}")
    if result.prototype_distance is not None and result.prototype_threshold is not None:
        diagnostics.append(
            f"feature_distance={result.prototype_distance:.3f}/{result.prototype_threshold:.3f}"
        )
    if result.metal_detected is not None:
        diagnostics.append(f"metal_sensor={'yes' if result.metal_detected else 'no'}")
    if diagnostics:
        print("Safety checks: " + ", ".join(diagnostics))
    if result.decision.accepted and result.decision.route in SAFE_LID_ROUTES:
        suffix = " [dry run]" if dry_run else ""
        print(
            f"OPEN {result.decision.route}: {result.decision.confidence:.1%} "
            f"(margin {result.decision.margin:.1%}){suffix}"
        )
    else:
        print(f"REJECTED, all lids closed: {result.decision.reason}")


def _apply_decision(
    lids: Any,
    result: AnalysisResult,
    *,
    hold_open: float,
    dry_run: bool,
) -> float | None:
    """Actuate a safe route and return its display-only close deadline.

    The controller closes on its own timer, independently of camera capture or
    UI progress. The context manager also requests closure on exit.
    """

    route = result.decision.route
    if not result.decision.accepted or route not in SAFE_LID_ROUTES or route == "other":
        lids.close_all()
        return None
    if dry_run:
        lids.open_temporarily(route, seconds=0)
        return None
    lids.open_timed(route, hold_open)
    return time.monotonic() + hold_open


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.i2c_bus is None:
        raise ValueError("live hardware mode requires an explicit I2C bus")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the target BSP
        raise RuntimeError("OpenCV Python bindings are required for the live demo") from exc

    delegate = None if args.delegate.lower() == "none" else args.delegate
    classifier = WasteClassifier(
        args.model,
        args.labels,
        delegate=delegate,
        threads=args.threads,
        metadata_path=args.metadata,
    )
    labels = set(classifier.labels)
    middle_labels = labels & {"paper", "general"}
    expected = {"plastic", "metal"} | middle_labels
    if len(middle_labels) != 1 or labels not in (expected, expected | {"other"}):
        raise RuntimeError(
            "The demo requires plastic, metal, exactly one of general/paper, "
            "and optionally other"
        )
    supports_unknown_rejection = getattr(
        classifier, "supports_unknown_rejection", "other" in labels
    )
    if not supports_unknown_rejection:
        if not args.dry_run:
            raise RuntimeError(
                "Live hardware requires a model with an 'other' class; "
                "use --dry-run with this three-class model"
            )
        print(
            "WARNING: this model has no 'other' class. Treat results as classification-only; "
            "unsupported items may be assigned to a known class."
        )
    validity_threshold = (
        args.validity_threshold
        if args.validity_threshold is not None
        else getattr(classifier, "recommended_validity_threshold", 0.5)
    )
    middle_label = next(iter(middle_labels))
    correction_keys = dict(CORRECTION_KEYS)
    correction_keys[ord("g")] = middle_label
    backend = _backend_name(classifier)
    print(classifier.describe())
    print("SPACE analyze | A arm/cancel auto | P/G/M/O save correction | Q/ESC quit")
    print(f"Starting in {'DRY RUN' if args.dry_run else 'LIVE HARDWARE'} mode")

    from ecosort_hw.lids import LidController
    from ecosort_hw.sensors import DigitalMetalSensor

    metal_sensor = (
        DigitalMetalSensor(args.metal_sensor_path, active_low=args.metal_sensor_active_low)
        if args.metal_sensor_path is not None
        else None
    )

    def analyze_buffer(frames: tuple[np.ndarray, ...]) -> AnalysisResult:
        return analyze_frames(
            classifier,
            frames,
            confidence_threshold=args.threshold,
            margin_threshold=args.margin,
            agreement_threshold=args.agreement,
            validity_threshold=validity_threshold,
            metal_detected=metal_sensor.read() if metal_sensor is not None else None,
        )

    frame_buffer: deque[np.ndarray] = deque(maxlen=args.average_frames)
    last_result: AnalysisResult | None = None
    last_auto_analysis = 0.0
    auto_mode = False
    lid_close_deadline: float | None = None
    active_route: str | None = None
    status = "Place one item inside the square"

    bus_number = args.i2c_bus if args.i2c_bus is not None else 1
    lids_controller = (
        LidController.from_config_file(
            args.hardware_config,
            bus=bus_number,
            address=args.i2c_address,
            dry_run=args.dry_run,
        )
        if args.hardware_config
        else LidController.from_defaults(
            bus=bus_number,
            address=args.i2c_address,
            dry_run=args.dry_run,
        )
    )

    with lids_controller as lids:
        with OpenCVCamera(
            args.camera,
            width=args.camera_width,
            height=args.camera_height,
        ) as camera:
            try:
                while True:
                    lids.raise_pending_error()
                    rgb, bgr = camera.read()
                    roi_bounds = _centered_square(bgr, args.roi_scale)
                    left, top, right, bottom = roi_bounds
                    current_roi_rgb = rgb[top:bottom, left:right].copy()
                    current_roi_bgr = bgr[top:bottom, left:right].copy()
                    frame_buffer.append(current_roi_rgb)

                    now = time.monotonic()
                    if lid_close_deadline is not None:
                        if lids.active_lid is None:
                            lid_close_deadline = None
                            active_route = None
                            status = "Lid closed; remove the item before re-arming"
                        elif now >= lid_close_deadline:
                            status = f"{active_route} lid closing"
                        else:
                            remaining = lid_close_deadline - now
                            status = f"{active_route} lid open; closing in {remaining:.1f}s"
                    should_auto_analyze = (
                        auto_mode
                        and lid_close_deadline is None
                        and len(frame_buffer) == args.average_frames
                        and now - last_auto_analysis >= args.auto_interval
                    )
                    if should_auto_analyze:
                        try:
                            last_result = analyze_buffer(tuple(frame_buffer))
                            _print_result(last_result, dry_run=args.dry_run)
                            lid_close_deadline = _apply_decision(
                                lids,
                                last_result,
                                hold_open=args.hold_open,
                                dry_run=args.dry_run,
                            )
                            active_route = (
                                last_result.decision.route
                                if lid_close_deadline is not None
                                else None
                            )
                            # Auto is deliberately one-shot: never actuate or
                            # re-command closed servos repeatedly for one item.
                            auto_mode = False
                            if last_result.decision.accepted:
                                status = "Accepted; auto paused. Remove item, then press A to re-arm"
                            else:
                                status = "Rejected; auto paused. Reposition item, then press A"
                        except Exception as exc:
                            auto_mode = False
                            last_result = None
                            lids.close_all()
                            lid_close_deadline = None
                            active_route = None
                            status = f"Analysis error - lids closed: {exc}"
                            print(status)
                        finally:
                            frame_buffer.clear()
                            last_auto_analysis = time.monotonic()

                    preview = bgr.copy()
                    _draw_overlay(
                        cv2,
                        preview,
                        roi_bounds,
                        backend=backend,
                        dry_run=args.dry_run,
                        auto_mode=auto_mode,
                        buffered_frames=len(frame_buffer),
                        required_frames=args.average_frames,
                        result=last_result,
                        status=status,
                    )
                    cv2.imshow("EcoSort Advanced Live Demo", preview)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break
                    if key in (ord("a"), ord("A")):
                        if lid_close_deadline is not None:
                            status = "Wait for the open lid to close before re-arming"
                            continue
                        auto_mode = not auto_mode
                        frame_buffer.clear()
                        last_auto_analysis = time.monotonic()
                        status = f"Auto {'ARMED - hold item steady' if auto_mode else 'CANCELLED'}"
                        continue
                    if key == 32:
                        # Manual analysis consumes an armed one-shot too.
                        auto_mode = False
                        if lid_close_deadline is not None:
                            status = "Wait for the open lid to close before another analysis"
                            continue
                        if len(frame_buffer) < args.average_frames:
                            status = (
                                f"Collecting frames: {len(frame_buffer)}/{args.average_frames}"
                            )
                            continue
                        try:
                            last_result = analyze_buffer(tuple(frame_buffer))
                            _print_result(last_result, dry_run=args.dry_run)
                            lid_close_deadline = _apply_decision(
                                lids,
                                last_result,
                                hold_open=args.hold_open,
                                dry_run=args.dry_run,
                            )
                            active_route = (
                                last_result.decision.route
                                if lid_close_deadline is not None
                                else None
                            )
                            status = (
                                f"{active_route} lid open"
                                if lid_close_deadline is not None
                                else "Manual analysis complete"
                            )
                        except Exception as exc:
                            last_result = None
                            lids.close_all()
                            lid_close_deadline = None
                            active_route = None
                            status = f"Analysis error - lids closed: {exc}"
                            print(status)
                        finally:
                            frame_buffer.clear()
                            last_auto_analysis = time.monotonic()
                        continue

                    normalized_key = ord(chr(key).lower()) if 0 <= key <= 255 else key
                    correction_label = correction_keys.get(normalized_key)
                    if correction_label is not None:
                        try:
                            path = _save_correction(
                                cv2,
                                current_roi_bgr,
                                args.corrections_dir,
                                correction_label,
                            )
                            status = f"Saved {correction_label}: {path}"
                            print(status)
                        except OSError as exc:
                            status = f"Could not save correction: {exc}"
                            print(status)
            finally:
                cv2.destroyAllWindows()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()
