"""Live visual preview with the compact mask or a laptop-only SAM teacher.

This viewer never actuates hardware. It can render the predicted item on white
so the segmentation head is easy to inspect before board integration.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .full_model import FullWasteModel
from .held_object import HeldObjectWorkerClient
from .mask_refinement import refine_binary_mask, smooth_mask_probability


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the full EcoSort model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts_full/full_waste_model_int8.tflite"),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("artifacts_full/labels.txt")
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--roi-scale", type=float, default=0.68)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument(
        "--raw-mask",
        action="store_true",
        help="show the unfiltered student mask (useful for model debugging)",
    )
    parser.add_argument(
        "--mask-smoothing",
        type=float,
        default=0.72,
        help="weight of the current frame in temporal mask smoothing (0..1)",
    )
    parser.add_argument("--material-threshold", type=float)
    parser.add_argument("--validity-threshold", type=float)
    parser.add_argument(
        "--held-object",
        action="store_true",
        help="laptop CUDA: detect a held item and outline it with EfficientViT-SAM-L0",
    )
    parser.add_argument(
        "--teacher-python", type=Path, default=Path(".teacher-venv/Scripts/python.exe")
    )
    parser.add_argument(
        "--held-object-cache", type=Path, default=Path("teacher_models")
    )
    parser.add_argument(
        "--efficientvit-repo", type=Path, default=Path("external/efficientvit")
    )
    parser.add_argument(
        "--efficientvit-checkpoint",
        type=Path,
        default=Path(
            "external/efficientvit/assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt"
        ),
    )
    parser.add_argument("--held-device", default="cuda")
    parser.add_argument("--object-model", default="yolov8n.pt")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--wrist-distance-ratio", type=float, default=0.22)
    parser.add_argument("--require-wrist", action="store_true")
    parser.add_argument(
        "--view",
        choices=("white", "overlay"),
        default="white",
        help="replace the ROI background with white or draw a transparent mask",
    )
    parser.add_argument(
        "--delegate",
        default="auto",
        help="auto, none, or a TensorFlow Lite delegate library path",
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if not 0.2 <= args.roi_scale <= 1.0:
        parser.error("--roi-scale must be between 0.2 and 1.0")
    if not 0.0 < args.mask_threshold < 1.0:
        parser.error("--mask-threshold must be between 0 and 1")
    if not 0.0 <= args.mask_smoothing <= 1.0:
        parser.error("--mask-smoothing must be between 0 and 1")
    if not 0.0 < args.detector_threshold < 1.0:
        parser.error("--detector-threshold must be between 0 and 1")
    if args.wrist_distance_ratio <= 0:
        parser.error("--wrist-distance-ratio must be positive")
    if args.camera_width is not None and args.camera_width < 160:
        parser.error("--camera-width must be at least 160")
    if args.camera_height is not None and args.camera_height < 120:
        parser.error("--camera-height must be at least 120")
    return args


def _put(
    cv2: object,
    frame: np.ndarray,
    text: str,
    row: int,
    *,
    color: tuple[int, int, int] = (240, 240, 240),
    scale: float = 0.67,
) -> None:
    cv2.putText(
        frame,
        text,
        (18, 30 + row * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (18, 30 + row * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _padded_box(
    shape: tuple[int, ...], box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    left, top, right, bottom = box
    pad_x = max(8, round((right - left) * 0.12))
    pad_y = max(8, round((bottom - top) * 0.12))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid detected item box: {box}")
    return left, top, right, bottom


def run(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required. Install requirements-camera.txt first."
        ) from exc

    delegate = None if args.delegate.lower() == "none" else args.delegate
    model = FullWasteModel(
        args.model,
        args.labels,
        delegate=delegate,
        metadata_path=args.metadata,
    )
    material_threshold = (
        args.material_threshold
        if args.material_threshold is not None
        else model.material_threshold
    )
    validity_threshold = (
        args.validity_threshold
        if args.validity_threshold is not None
        else model.validity_threshold
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    if args.camera_width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height is not None:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    score_history: deque[np.ndarray] = deque(maxlen=args.frames)
    validity_history: deque[float] = deque(maxlen=args.frames)
    previous_mask_probability: np.ndarray | None = None
    previous = time.perf_counter()
    fps = 0.0
    print(model.describe())
    print("Visual dry run only: no lid or actuator is controlled.")

    project_root = Path(__file__).resolve().parents[1]
    try:
        with ExitStack() as stack:
            held_worker = (
                stack.enter_context(
                    HeldObjectWorkerClient(
                        python_executable=args.teacher_python,
                        worker_script=project_root
                        / "scripts"
                        / "held_object_worker.py",
                        model_cache=args.held_object_cache,
                        efficientvit_repo=args.efficientvit_repo,
                        checkpoint=args.efficientvit_checkpoint,
                        device=args.held_device,
                        object_model=args.object_model,
                        pose_model=args.pose_model,
                        detector_threshold=args.detector_threshold,
                        wrist_distance_ratio=args.wrist_distance_ratio,
                        allow_center_fallback=not args.require_wrist,
                    )
                )
                if args.held_object
                else None
            )
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                height, width = frame.shape[:2]
                side = int(min(height, width) * args.roi_scale)
                left = (width - side) // 2
                top = (height - side) // 2
                right, bottom = left + side, top + side
                prediction = None
                object_note = ""
                if held_worker is not None:
                    selection = held_worker.analyze_bgr(
                        frame, (left, top, right, bottom)
                    )
                    if selection.accepted:
                        if selection.mask is None or selection.bbox_xyxy is None:
                            raise RuntimeError(
                                "Held-object worker omitted its mask or box"
                            )
                        if selection.mask.shape != (height, width):
                            raise RuntimeError(
                                "SAM mask does not match the camera frame"
                            )
                        box = _padded_box(frame.shape, selection.bbox_xyxy)
                        x1, y1, x2, y2 = box
                        crop = frame[y1:y2, x1:x2]
                        prediction = model.predict_rgb(
                            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        )
                        selected = selection.mask.astype(bool)
                        region = frame
                        origin = (0, 0)
                        rectangle = box
                        object_note = (
                            f"SAM: {selection.object_name or 'item'} "
                            f"{(selection.mask_confidence or 0.0):.0%} mask"
                        )
                        # A moving detector box should not inherit scores from
                        # earlier camera crops; each SAM outline is current-frame.
                        score_history.clear()
                        validity_history.clear()
                        previous_mask_probability = None
                    else:
                        selected = np.zeros((height, width), dtype=bool)
                        region = frame
                        origin = (0, 0)
                        rectangle = (left, top, right, bottom)
                        object_note = f"No held item: {selection.reason}"
                        score_history.clear()
                        validity_history.clear()
                        previous_mask_probability = None
                else:
                    region = frame[top:bottom, left:right]
                    prediction = model.predict_rgb(
                        cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    )
                    mask = cv2.resize(
                        prediction.object_mask,
                        (side, side),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask = smooth_mask_probability(
                        mask,
                        previous_mask_probability,
                        current_weight=args.mask_smoothing,
                    )
                    previous_mask_probability = mask
                    selected = (
                        mask >= args.mask_threshold
                        if args.raw_mask
                        else refine_binary_mask(
                            cv2,
                            mask,
                            threshold=args.mask_threshold,
                        )
                    )
                    origin = (left, top)
                    rectangle = (left, top, right, bottom)
                    object_note = (
                        "INT8 raw student mask"
                        if args.raw_mask
                        else "INT8 student mask + NXP-safe cleanup"
                    )

                if prediction is not None:
                    score_history.append(
                        np.asarray(
                            [prediction.scores[label] for label in model.labels],
                            dtype=np.float32,
                        )
                    )
                    validity_history.append(prediction.supported_probability)
                    scores = np.mean(np.stack(score_history), axis=0)
                    supported = float(np.mean(validity_history))
                    material_index = int(np.argmax(scores))
                    material = model.labels[material_index]
                    confidence = float(scores[material_index])

                    if args.view == "white":
                        displayed = np.full_like(region, 255)
                        displayed[selected] = region[selected]
                        if held_worker is None:
                            frame[top:bottom, left:right] = displayed
                        else:
                            frame[:] = displayed
                    else:
                        tint = np.zeros_like(region)
                        tint[:] = (80, 220, 80)
                        blended = cv2.addWeighted(region, 0.58, tint, 0.42, 0)
                        region[selected] = blended[selected]

                    contour_mask = selected.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    ox, oy = origin
                    shifted = [contour + np.array([[[ox, oy]]]) for contour in contours]
                    cv2.drawContours(frame, shifted, -1, (60, 240, 80), 2)
                else:
                    material = "NO ITEM"
                    confidence = 0.0
                    supported = 0.0

                x1, y1, x2, y2 = rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 210, 90), 1)
                route_preview = model.metadata.get("taxonomy") == "disposal"
                safe = (
                    prediction is not None
                    and held_worker is None
                    and not route_preview
                    and supported >= validity_threshold
                    and confidence >= material_threshold
                )
                status = (
                    "VISUAL PREVIEW ONLY"
                    if route_preview
                    else f"READY: {material.upper()}"
                    if safe
                    else "REJECT / WAIT"
                )
                status_color = (
                    (60, 210, 255)
                    if route_preview
                    else (60, 240, 80)
                    if safe
                    else (60, 80, 255)
                )
                _put(
                    cv2,
                    frame,
                    "EcoSort FULL MODEL - VISUAL DRY RUN",
                    0,
                    color=(80, 255, 120),
                )
                label_name = "Route" if route_preview else "Material"
                _put(
                    cv2, frame, f"{label_name}: {material.upper()}  {confidence:.1%}", 1
                )
                _put(
                    cv2,
                    frame,
                    "Unknown/reject gate: NOT TRAINED"
                    if route_preview
                    else f"Supported: {supported:.1%}",
                    2,
                )
                _put(cv2, frame, f"Decision: {status}", 3, color=status_color)
                _put(
                    cv2,
                    frame,
                    f"Mask pixels: {selected.mean():.1%} | {fps:.1f} FPS",
                    4,
                )
                _put(cv2, frame, object_note, 5, scale=0.52)
                _put(cv2, frame, "Q / ESC: quit   V: switch view", 6, scale=0.52)

                now = time.perf_counter()
                instantaneous = 1.0 / max(now - previous, 1e-6)
                fps = instantaneous if fps == 0 else fps * 0.85 + instantaneous * 0.15
                previous = now
                cv2.imshow("EcoSort full model", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
                    args.view = "overlay" if args.view == "white" else "white"
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    run(arguments())


if __name__ == "__main__":
    main()
