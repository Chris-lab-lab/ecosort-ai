from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2


KEY_TO_LABEL = {
    ord("e"): "empty",
    ord("E"): "empty",
    ord("h"): "half-full",
    ord("H"): "half-full",
    ord("f"): "full",
    ord("F"): "full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled webcam images of a specific waste bin.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def center_square(frame):
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side], (left, top, side)


def main() -> None:
    args = parse_args()
    for label in ("empty", "half-full", "full"):
        (args.output_dir / label).mkdir(parents=True, exist_ok=True)

    counts = Counter(
        {
            label: len(list((args.output_dir / label).glob("*.jpg")))
            for label in ("empty", "half-full", "full")
        }
    )
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Try a different -Camera number.")

    notice = ""
    notice_until = 0.0
    print("E: save EMPTY | H: save HALF-FULL | F: save FULL | Q: quit")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            crop, (left, top, side) = center_square(frame)
            preview = frame.copy()
            cv2.rectangle(preview, (left, top), (left + side, top + side), (0, 255, 0), 2)
            cv2.rectangle(preview, (8, 8), (preview.shape[1] - 8, 100), (0, 0, 0), -1)
            cv2.putText(
                preview,
                "E=empty  H=half-full  F=full  Q=quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            count_text = f"Saved: empty {counts['empty']} | half {counts['half-full']} | full {counts['full']}"
            cv2.putText(preview, count_text, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if time.time() < notice_until:
                cv2.putText(
                    preview,
                    notice,
                    (20, preview.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Collect custom waste-bin training images", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in KEY_TO_LABEL:
                label = KEY_TO_LABEL[key]
                filename = f"{label}_{time.time_ns()}.jpg"
                path = args.output_dir / label / filename
                cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                counts[label] += 1
                notice = f"Saved {label} image #{counts[label]}"
                notice_until = time.time() + 1.0
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(dict(counts))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
