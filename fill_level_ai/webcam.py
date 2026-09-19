from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import cv2
from PIL import Image

from inference import default_model_path, load_model, predict_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test robust trash-bin fill detection with a webcam.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument(
        "--edge-api-url",
        help="Publish stable fill states to an EcoSort edge API, for example http://BOARD_IP:8080.",
    )
    parser.add_argument("--bin", choices=("plastic", "metal", "general"), default="plastic")
    parser.add_argument("--stable-predictions", type=int, default=3)
    parser.add_argument("--publish-heartbeat-seconds", type=float, default=10.0)
    return parser.parse_args()


def publish_fill_state(
    edge_api_url: str,
    category: str,
    fill_state: str,
    confidence: float,
) -> None:
    body = json.dumps(
        {
            "fill_state": fill_state,
            "source": "webcam",
            "confidence": confidence,
        }
    ).encode("utf-8")
    request = Request(
        f"{edge_api_url.rstrip('/')}/api/bins/{category}/fill-state",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"Edge API returned HTTP {response.status}")


def main() -> None:
    args = parse_args()
    if args.stable_predictions < 1:
        raise ValueError("--stable-predictions must be at least 1")
    if args.publish_heartbeat_seconds <= 0:
        raise ValueError("--publish-heartbeat-seconds must be positive")
    model = load_model(args.model)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Try --camera 1.")

    result = {"prediction": "starting..."}
    frame_number = 0
    candidate_state = ""
    candidate_count = 0
    published_state = ""
    last_published_at = 0.0
    publish_status = "API publishing disabled"
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
                predicted_state = result["prediction"]
                if predicted_state == candidate_state:
                    candidate_count += 1
                else:
                    candidate_state = predicted_state
                    candidate_count = 1

                if (
                    args.edge_api_url
                    and candidate_count >= args.stable_predictions
                    and (
                        candidate_state != published_state
                        or time.monotonic() - last_published_at
                        >= args.publish_heartbeat_seconds
                    )
                ):
                    try:
                        publish_fill_state(
                            args.edge_api_url,
                            args.bin,
                            candidate_state,
                            float(result["confidence"]),
                        )
                        published_state = candidate_state
                        last_published_at = time.monotonic()
                        publish_status = f"Published {candidate_state} to {args.bin} bin"
                        print(publish_status)
                    except Exception as exc:
                        publish_status = f"API publish failed: {exc}"
                        print(publish_status)
            frame_number += 1

            label = result["prediction"].upper()
            height, width = frame.shape[:2]
            side = int(min(height, width) * 0.88)
            top, left = (height - side) // 2, (width - side) // 2
            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (left + side, top + side), (0, 255, 255), 2)
            cv2.rectangle(frame, (8, 8), (min(width - 8, 720), 100), (0, 0, 0), thickness=-1)
            cv2.putText(frame, label, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, "Keep the full bin inside the yellow guide", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.putText(frame, publish_status, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, "Q: quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Robust waste-bin fill AI", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
