"""Held-object selection and the live GPU-worker client.

The main live demo intentionally keeps TensorFlow and PyTorch in separate
virtual environments. Frames are sent to a persistent worker as JPEGs so the
large detector, pose, and EfficientViT-SAM models are loaded only once.
"""

from __future__ import annotations

import base64
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ObjectCandidate:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class HeldObjectSelection:
    accepted: bool
    reason: str
    object_name: str | None = None
    detector_confidence: float | None = None
    mask_confidence: float | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    hand_present: bool = False
    selection_source: str | None = None
    classifier_rgb: np.ndarray | None = None
    mask: np.ndarray | None = None


def _point_box_distance(
    point: tuple[float, float], box: tuple[int, int, int, int]
) -> float:
    x, y = point
    left, top, right, bottom = box
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return math.hypot(dx, dy)


def select_held_candidate(
    candidates: Iterable[ObjectCandidate],
    wrists: Iterable[tuple[float, float]],
    image_shape: tuple[int, int] | tuple[int, int, int],
    roi_xyxy: tuple[int, int, int, int],
    *,
    max_wrist_distance_ratio: float = 0.22,
    allow_center_fallback: bool = True,
) -> tuple[ObjectCandidate | None, str]:
    """Choose the detected non-person object nearest a visible wrist.

    When pose estimation cannot see a wrist, a conservative center-ROI
    fallback keeps the system usable for an item placed in the presentation
    square. A visible wrist never silently falls back to an unrelated object.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < max_wrist_distance_ratio <= 1.0:
        raise ValueError("max wrist distance ratio must be between 0 and 1")

    left, top, right, bottom = roi_xyxy
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("ROI is outside the image")

    usable: list[ObjectCandidate] = []
    for candidate in candidates:
        box = candidate.bbox_xyxy
        if candidate.label.strip().lower() == "person":
            continue
        if (
            not math.isfinite(candidate.confidence)
            or not 0.0 <= candidate.confidence <= 1.0
        ):
            continue
        box_left, box_top, box_right, box_bottom = box
        if not (
            0 <= box_left < box_right <= width and 0 <= box_top < box_bottom <= height
        ):
            continue
        area_ratio = ((box_right - box_left) * (box_bottom - box_top)) / (
            width * height
        )
        if 0.001 <= area_ratio <= 0.80:
            usable.append(candidate)

    if not usable:
        return None, "no supported object detected"

    wrist_points = tuple(wrists)
    if wrist_points:
        maximum_distance = math.hypot(width, height) * max_wrist_distance_ratio
        ranked: list[tuple[float, ObjectCandidate]] = []
        for candidate in usable:
            distance = min(
                _point_box_distance(point, candidate.bbox_xyxy)
                for point in wrist_points
            )
            if distance <= maximum_distance:
                proximity = 1.0 - distance / maximum_distance
                ranked.append((candidate.confidence + proximity, candidate))
        if ranked:
            return max(ranked, key=lambda item: item[0])[1], "wrist"
        return None, "objects detected, but none is near a visible wrist"

    if not allow_center_fallback:
        return None, "no reliable wrist detected"

    roi_center_x = (left + right) / 2.0
    roi_center_y = (top + bottom) / 2.0
    roi_diagonal = max(1.0, math.hypot(right - left, bottom - top))
    ranked_center: list[tuple[float, ObjectCandidate]] = []
    for candidate in usable:
        box_left, box_top, box_right, box_bottom = candidate.bbox_xyxy
        center_x = (box_left + box_right) / 2.0
        center_y = (box_top + box_bottom) / 2.0
        if not (left <= center_x <= right and top <= center_y <= bottom):
            continue
        center_distance = (
            math.hypot(center_x - roi_center_x, center_y - roi_center_y) / roi_diagonal
        )
        ranked_center.append((candidate.confidence - center_distance * 0.35, candidate))
    if not ranked_center:
        return None, "place a detected object inside the center square"
    return max(ranked_center, key=lambda item: item[0])[1], "center"


class HeldObjectWorkerClient:
    """Persistent JSON-lines client for ``scripts/held_object_worker.py``."""

    def __init__(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        model_cache: Path,
        efficientvit_repo: Path,
        checkpoint: Path,
        device: str,
        object_model: str,
        pose_model: str,
        detector_threshold: float,
        wrist_distance_ratio: float,
        allow_center_fallback: bool,
    ) -> None:
        self.python_executable = python_executable.resolve()
        self.worker_script = worker_script.resolve()
        self.model_cache = model_cache.resolve()
        self.efficientvit_repo = efficientvit_repo.resolve()
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.object_model = object_model
        self.pose_model = pose_model
        self.detector_threshold = detector_threshold
        self.wrist_distance_ratio = wrist_distance_ratio
        self.allow_center_fallback = allow_center_fallback
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "HeldObjectWorkerClient":
        if not self.python_executable.is_file():
            raise RuntimeError(
                f"Held-object Python environment not found: {self.python_executable}"
            )
        if not self.worker_script.is_file():
            raise RuntimeError(
                f"Held-object worker script not found: {self.worker_script}"
            )
        if not self.efficientvit_repo.is_dir():
            raise RuntimeError(
                f"EfficientViT repository not found: {self.efficientvit_repo}"
            )
        if not self.checkpoint.is_file():
            raise RuntimeError(f"EfficientViT checkpoint not found: {self.checkpoint}")
        self.model_cache.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python_executable),
            "-u",
            str(self.worker_script),
            "--efficientvit-repo",
            str(self.efficientvit_repo),
            "--checkpoint",
            str(self.checkpoint),
            "--device",
            self.device,
            "--object-model",
            self.object_model,
            "--pose-model",
            self.pose_model,
            "--detector-threshold",
            str(self.detector_threshold),
            "--wrist-distance-ratio",
            str(self.wrist_distance_ratio),
        ]
        if self.allow_center_fallback:
            command.append("--allow-center-fallback")
        self.process = subprocess.Popen(
            command,
            cwd=self.model_cache,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        response = self._read_response()
        if response.get("type") != "ready":
            self.close()
            raise RuntimeError(
                str(response.get("error", "held-object worker did not become ready"))
            )
        return self

    def _read_response(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("held-object worker is not running")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(
                f"held-object worker stopped unexpectedly (exit code {code})"
            )
        response = json.loads(line)
        if response.get("type") == "fatal":
            raise RuntimeError(str(response.get("error", "held-object worker failed")))
        return response

    def analyze_bgr(
        self,
        bgr: np.ndarray,
        roi_xyxy: tuple[int, int, int, int],
    ) -> HeldObjectSelection:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - camera dependency
            raise RuntimeError("OpenCV is required for held-object mode") from exc
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("held-object worker is not running")
        success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise RuntimeError("could not encode camera frame for held-object worker")
        request = {
            "type": "frame",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "roi": list(roi_xyxy),
        }
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        if response.get("type") != "result":
            raise RuntimeError("held-object worker returned an invalid response")
        if not response.get("accepted", False):
            return HeldObjectSelection(
                accepted=False,
                reason=str(response.get("reason", "no held object detected")),
                hand_present=bool(response.get("hand_present", False)),
            )

        crop_bytes = base64.b64decode(response["crop"])
        crop_bgr = cv2.imdecode(
            np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        mask_bytes = base64.b64decode(response["mask"])
        mask = cv2.imdecode(
            np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if crop_bgr is None or mask is None:
            raise RuntimeError("held-object worker returned corrupt image data")
        return HeldObjectSelection(
            accepted=True,
            reason="accepted",
            object_name=str(response["object_name"]),
            detector_confidence=float(response["detector_confidence"]),
            mask_confidence=float(response["mask_confidence"]),
            bbox_xyxy=tuple(int(value) for value in response["bbox"]),
            hand_present=bool(response.get("hand_present", False)),
            selection_source=str(response.get("selection_source", "unknown")),
            classifier_rgb=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
            mask=mask > 0,
        )

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write('{"type":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def __exit__(self, *_: object) -> None:
        self.close()
