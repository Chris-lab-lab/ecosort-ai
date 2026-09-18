"""Persistent GPU worker for held-object detection and EfficientViT-SAM.

This process is launched by ``ecosort_ai.live_demo`` using the separate
PyTorch teacher environment. Standard output is reserved for a JSON-lines
protocol; model logs are redirected to standard error.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ecosort_ai.efficientvit_compat import install_triton_rms_norm_fallback
from ecosort_ai.held_object import ObjectCandidate, select_held_candidate


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoSort held-object GPU worker")
    parser.add_argument("--efficientvit-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object-model", default="yolov8n.pt")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--wrist-distance-ratio", type=float, default=0.22)
    parser.add_argument("--allow-center-fallback", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.detector_threshold <= 1.0:
        parser.error("--detector-threshold must be between 0 and 1")
    if not 0.0 < args.wrist_distance_ratio <= 1.0:
        parser.error("--wrist-distance-ratio must be between 0 and 1")
    return args


def _encode_image(
    image: np.ndarray, extension: str, parameters: list[int] | None = None
) -> str:
    success, encoded = cv2.imencode(extension, image, parameters or [])
    if not success:
        raise RuntimeError(f"could not encode {extension} worker output")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _clamp_box(
    box: np.ndarray, width: int, height: int
) -> tuple[int, int, int, int] | None:
    left = max(0, min(width - 1, int(round(float(box[0])))))
    top = max(0, min(height - 1, int(round(float(box[1])))))
    right = max(left + 1, min(width, int(round(float(box[2])))))
    bottom = max(top + 1, min(height, int(round(float(box[3])))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


class HeldObjectPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from ultralytics import YOLO

        repository = args.efficientvit_repo.resolve()
        checkpoint = args.checkpoint.resolve()
        if not (repository / "efficientvit" / "sam_model_zoo.py").is_file():
            raise RuntimeError(f"EfficientViT repository is incomplete: {repository}")
        if not checkpoint.is_file():
            raise RuntimeError(f"EfficientViT checkpoint was not found: {checkpoint}")
        sys.path.insert(0, str(repository))
        self.using_triton_fallback = install_triton_rms_norm_fallback(torch)
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access the NVIDIA GPU"
            )
        self.device = args.device
        self.ultralytics_device: str | int = (
            int(args.device.split(":", 1)[1])
            if args.device.startswith("cuda:")
            else 0
            if args.device == "cuda"
            else args.device
        )
        self.detector = YOLO(args.object_model)
        self.pose = YOLO(args.pose_model)
        model = (
            create_efficientvit_sam_model(
                name="efficientvit-sam-l0",
                pretrained=True,
                weight_url=str(checkpoint),
            )
            .to(args.device)
            .eval()
        )
        self.predictor = EfficientViTSamPredictor(model)
        self.detector_threshold = args.detector_threshold
        self.wrist_distance_ratio = args.wrist_distance_ratio
        self.allow_center_fallback = args.allow_center_fallback

    def _detections(self, bgr: np.ndarray) -> list[ObjectCandidate]:
        result = self.detector.predict(
            bgr,
            conf=self.detector_threshold,
            device=self.ultralytics_device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        height, width = bgr.shape[:2]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        candidates: list[ObjectCandidate] = []
        for raw_box, confidence, class_id in zip(
            boxes, confidences, classes, strict=True
        ):
            box = _clamp_box(raw_box, width, height)
            if box is None:
                continue
            label = str(result.names[int(class_id)])
            candidates.append(ObjectCandidate(label, float(confidence), box))
        return candidates

    def _wrists(self, bgr: np.ndarray) -> list[tuple[float, float]]:
        result = self.pose.predict(
            bgr,
            conf=max(0.20, self.detector_threshold),
            device=self.ultralytics_device,
            verbose=False,
        )[0]
        if result.keypoints is None or result.keypoints.data is None:
            return []
        keypoints = result.keypoints.data.detach().cpu().numpy()
        wrists: list[tuple[float, float]] = []
        for person in keypoints:
            for wrist_index in (9, 10):
                if wrist_index >= len(person):
                    continue
                point = person[wrist_index]
                confidence = float(point[2]) if len(point) >= 3 else 1.0
                if confidence >= 0.25 and float(point[0]) > 0 and float(point[1]) > 0:
                    wrists.append((float(point[0]), float(point[1])))
        return wrists

    def _segment(
        self,
        bgr: np.ndarray,
        candidate: ObjectCandidate,
    ) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(candidate.bbox_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        if len(masks) == 0:
            raise RuntimeError("EfficientViT-SAM returned no mask")
        box_left, box_top, box_right, box_bottom = candidate.bbox_xyxy
        box_area = max(1, (box_right - box_left) * (box_bottom - box_top))
        box_center = ((box_left + box_right) // 2, (box_top + box_bottom) // 2)
        ranked: list[tuple[float, np.ndarray, float]] = []
        for mask, score in zip(masks, scores, strict=True):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != bgr.shape[:2] or not mask[box_center[1], box_center[0]]:
                continue
            area_ratio = float(mask.sum()) / box_area
            if not 0.08 <= area_ratio <= 4.0:
                continue
            quality = float(score) - abs(math.log(max(area_ratio, 1e-6))) * 0.04
            ranked.append((quality, mask, float(score)))
        if not ranked:
            raise RuntimeError(
                "EfficientViT-SAM masks did not match the detected object"
            )
        _, mask, score = max(ranked, key=lambda item: item[0])
        return mask, score

    @staticmethod
    def _isolated_crop(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rows, columns = np.nonzero(mask)
        if not len(rows):
            raise RuntimeError("segmentation mask is empty")
        left, right = int(columns.min()), int(columns.max()) + 1
        top, bottom = int(rows.min()), int(rows.max()) + 1
        pad_x = max(2, int((right - left) * 0.06))
        pad_y = max(2, int((bottom - top) * 0.06))
        left, top = max(0, left - pad_x), max(0, top - pad_y)
        right, bottom = (
            min(bgr.shape[1], right + pad_x),
            min(bgr.shape[0], bottom + pad_y),
        )
        background_pixels = bgr[~mask]
        fill = (
            np.median(background_pixels, axis=0).astype(np.uint8)
            if len(background_pixels)
            else np.asarray((127, 127, 127), dtype=np.uint8)
        )
        isolated = np.empty_like(bgr)
        isolated[:] = fill
        isolated[mask] = bgr[mask]
        return isolated[top:bottom, left:right]

    def analyze(
        self, bgr: np.ndarray, roi: tuple[int, int, int, int]
    ) -> dict[str, Any]:
        candidates = self._detections(bgr)
        wrists = self._wrists(bgr)
        selected, source = select_held_candidate(
            candidates,
            wrists,
            bgr.shape,
            roi,
            max_wrist_distance_ratio=self.wrist_distance_ratio,
            allow_center_fallback=self.allow_center_fallback,
        )
        if selected is None:
            return {
                "type": "result",
                "accepted": False,
                "reason": source,
                "hand_present": bool(wrists),
            }
        mask, mask_score = self._segment(bgr, selected)
        crop = self._isolated_crop(bgr, mask)
        return {
            "type": "result",
            "accepted": True,
            "reason": "accepted",
            "object_name": selected.label,
            "detector_confidence": selected.confidence,
            "mask_confidence": mask_score,
            "bbox": list(selected.bbox_xyxy),
            "hand_present": bool(wrists),
            "selection_source": source,
            "crop": _encode_image(crop, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92]),
            "mask": _encode_image((mask * 255).astype(np.uint8), ".png"),
        }


def _write(protocol: Any, payload: dict[str, Any]) -> None:
    protocol.write(json.dumps(payload, separators=(",", ":")) + "\n")
    protocol.flush()


def main() -> None:
    args = arguments()
    protocol = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            pipeline = HeldObjectPipeline(args)
        _write(
            protocol,
            {
                "type": "ready",
                "device": pipeline.device,
                "triton_fallback": pipeline.using_triton_fallback,
            },
        )
    except Exception as exc:
        _write(protocol, {"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        raise SystemExit(1) from exc

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "shutdown":
                break
            if request.get("type") != "frame":
                raise ValueError("unknown worker request")
            encoded = base64.b64decode(request["image"])
            bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("camera JPEG could not be decoded")
            roi = tuple(int(value) for value in request["roi"])
            if len(roi) != 4:
                raise ValueError("ROI must contain four coordinates")
            with redirect_stdout(sys.stderr):
                response = pipeline.analyze(bgr, roi)
            _write(protocol, response)
        except Exception as exc:
            _write(
                protocol,
                {
                    "type": "result",
                    "accepted": False,
                    "reason": f"worker error: {type(exc).__name__}: {exc}",
                    "hand_present": False,
                },
            )


if __name__ == "__main__":
    main()
