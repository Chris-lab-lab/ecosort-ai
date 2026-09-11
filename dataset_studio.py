"""One-window webcam studio for collecting all EcoSort classes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def centered_square(frame):
    """Return a centered square ROI and its (x1, y1, x2, y2) bounds."""

    height, width = frame.shape[:2]
    side = int(min(width, height) * 0.72)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return frame[y1 : y1 + side, x1 : x1 + side], (x1, y1, x1 + side, y1 + side)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the camera and collect plastic/middle/metal/other images."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default="data")
    parser.add_argument("--middle", choices=["general", "paper"], default="general")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    import cv2

    labels = ("plastic", args.middle, "metal", "other")
    destinations = {label: Path(args.output) / label for label in labels}
    for destination in destinations.values():
        destination.mkdir(parents=True, exist_ok=True)
    counts = {label: len(list(path.glob("*.jpg"))) for label, path in destinations.items()}
    key_to_label = {
        ord("1"): "plastic",
        ord("2"): args.middle,
        ord("3"): "metal",
        ord("0"): "other",
        ord("p"): "plastic",
        ord("g"): args.middle,
        ord("m"): "metal",
        ord("o"): "other",
    }

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera index {args.camera}. Try --camera 1 or reconnect the webcam."
        )

    last_message = "Place one object inside the square"
    print("Keys: 1=plastic, 2=middle, 3=metal, 0=other, Q=quit")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped returning frames")
            roi, (x1, y1, x2, y2) = centered_square(frame)
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (80, 230, 120), 3)
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 105), (24, 32, 30), -1)
            cv2.putText(
                preview,
                "1 PLASTIC   2 MIDDLE   3 METAL   0 OTHER   Q QUIT",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (240, 245, 240),
                2,
            )
            count_text = "   ".join(f"{label}: {counts[label]}" for label in labels)
            cv2.putText(
                preview,
                count_text,
                (18, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (120, 225, 170),
                2,
            )
            cv2.putText(
                preview,
                last_message,
                (18, 97),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (90, 210, 255),
                2,
            )
            cv2.imshow("EcoSort dataset studio", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            label = key_to_label.get(key)
            if label is not None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                target = destinations[label] / f"{label}_{stamp}.jpg"
                if not cv2.imwrite(str(target), roi):
                    raise OSError(f"Could not save {target}")
                counts[label] += 1
                last_message = f"Saved {label}: {target.name}"
                print(target)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
