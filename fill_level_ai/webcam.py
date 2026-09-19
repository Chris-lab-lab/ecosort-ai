from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from inference import default_model_path, load_model, predict_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test robust trash-bin fill detection with a webcam.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=default_model_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Try --camera 1.")

    result = {"prediction": "starting..."}
    frame_number = 0
    print(f"Using model: {args.model.resolve()}")
    print("Point the camera at the complete bin interior. Press Q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            if frame_number % 6 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = predict_pil(model, Image.fromarray(rgb))
            frame_number += 1

            label = result["prediction"].upper()
            height, width = frame.shape[:2]
            side = int(min(height, width) * 0.88)
            top, left = (height - side) // 2, (width - side) // 2
            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (left + side, top + side), (0, 255, 255), 2)
            cv2.rectangle(frame, (8, 8), (min(width - 8, 720), 78), (0, 0, 0), thickness=-1)
            cv2.putText(frame, label, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, "Keep the full bin inside the yellow guide", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.putText(frame, "Q: quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Robust waste-bin fill AI", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
