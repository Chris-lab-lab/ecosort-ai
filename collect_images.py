from __future__ import annotations

import argparse
import time
from pathlib import Path

from dataset_studio import centered_square


def main() -> None:
    p = argparse.ArgumentParser(description="Collect labeled webcam images for EcoSort")
    p.add_argument("label", choices=["plastic", "paper", "general", "metal", "other"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--output", default="data")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    import cv2

    destination = Path(args.output) / args.label
    destination.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    print(f"Saving to {destination}. SPACE saves; Q quits.")
    count = len(list(destination.glob("*.jpg")))
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame failed")
            roi, (x1, y1, x2, y2) = centered_square(frame)
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (80, 230, 120), 3)
            cv2.putText(preview, f"{args.label}: {count} images", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 255, 80), 2)
            cv2.imshow("EcoSort data collection", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:
                name = destination / f"{args.label}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(name), roi)
                count += 1
                print(name)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
