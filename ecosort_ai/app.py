from __future__ import annotations

import argparse
import time

from .camera import OpenCVCamera, read_rgb_image
from .decision import decide_route
from .model import WasteClassifier


def centered_roi(image, scale: float = 0.72):
    height, width = image.shape[:2]
    side = int(min(width, height) * scale)
    left = (width - side) // 2
    top = (height - side) // 2
    return image[top : top + side, left : left + side], (left, top, side)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EcoSort camera classifier and three-lid controller")
    p.add_argument("--model", required=True, help="TFLite or Vela-compiled TFLite model")
    p.add_argument("--labels", required=True, help="One label per line in model output order")
    p.add_argument("--image", help="Classify one image instead of opening a camera")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--delegate", default="auto", help="auto, none, or delegate .so path")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--margin", type=float, default=0.15)
    hardware = p.add_mutually_exclusive_group()
    hardware.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print lid actions without using I2C (default)",
    )
    hardware.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="redirects to the required multi-frame live_demo safety runner",
    )
    p.set_defaults(dry_run=True)
    p.add_argument(
        "--i2c-bus",
        type=int,
        help="Required for live hardware; discover it with: i2cdetect -l",
    )
    p.add_argument("--i2c-address", type=lambda value: int(value, 0), default=0x40)
    p.add_argument(
        "--hardware-config",
        help="JSON file with per-lid PCA9685 channels and calibrated pulses",
    )
    p.add_argument("--hold-open", type=float, default=4.0)
    p.add_argument("--headless", action="store_true", help="Use Enter instead of a preview window")
    return p


def classify_and_route(classifier: WasteClassifier, rgb, lids, args) -> None:
    prediction = classifier.predict_rgb(rgb)
    decision = decide_route(
        prediction.scores,
        confidence_threshold=args.threshold,
        margin_threshold=args.margin,
    )
    print("\nScores:")
    for label, score in sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {label:8s} {score:.1%}")

    if not decision.accepted:
        print(f"NO LID OPENED: {decision.reason}")
        return

    print(f"Accepted: {decision.route} ({decision.confidence:.1%})")
    # A laptop dry-run should display the result immediately rather than
    # pretending to hold a physical lid for several seconds.
    lids.open_temporarily(decision.route, seconds=0 if args.dry_run else args.hold_open)


def main() -> None:
    command = parser()
    args = command.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        command.error("--threshold must be between 0 and 1")
    if not 0.0 <= args.margin <= 1.0:
        command.error("--margin must be between 0 and 1")
    if args.hold_open < 0:
        command.error("--hold-open cannot be negative")
    if not args.dry_run and args.i2c_bus is None:
        command.error("--live requires --i2c-bus; find it with `i2cdetect -l`")
    if not args.dry_run:
        command.error(
            "physical lid control requires the multi-frame safety runner: "
            "use `python3 -m ecosort_ai.live_demo ... --live --i2c-bus BUS`"
        )
    delegate = None if args.delegate.lower() == "none" else args.delegate
    classifier = WasteClassifier(args.model, args.labels, delegate=delegate)
    labels = set(classifier.labels)
    middle = labels & {"general", "paper"}
    expected = {"plastic", "metal"} | middle
    if len(middle) != 1 or labels not in (expected, expected | {"other"}):
        raise RuntimeError(
            "The model must contain plastic, metal, exactly one of general/paper, "
            "and optionally other"
        )
    if "other" not in labels:
        print(
            "WARNING: this model has no 'other' class. Treat results as classification-only; "
            "unsupported items may be assigned to a known class."
        )
    print(classifier.describe())

    from ecosort_hw.lids import LidController

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
        lids.close_all()

        if args.image:
            classify_and_route(classifier, read_rgb_image(args.image), lids, args)
            return

        with OpenCVCamera(args.camera) as camera:
            if args.headless:
                print("Press Enter to classify an item; type q then Enter to quit.")
                while input("> ").strip().lower() != "q":
                    rgb, _ = camera.read()
                    roi_rgb, _ = centered_roi(rgb)
                    classify_and_route(classifier, roi_rgb, lids, args)
                return

            import cv2

            print("Preview: SPACE classifies, Q quits. Keep one item centered.")
            while True:
                rgb, bgr = camera.read()
                roi_rgb, (left, top, side) = centered_roi(rgb)
                cv2.rectangle(
                    bgr,
                    (left, top),
                    (left + side, top + side),
                    (80, 230, 120),
                    3,
                )
                cv2.putText(
                    bgr,
                    "SPACE: classify   Q: quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (80, 255, 80),
                    2,
                )
                cv2.imshow("EcoSort Twin", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    classify_and_route(classifier, roi_rgb, lids, args)
                    time.sleep(0.25)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
