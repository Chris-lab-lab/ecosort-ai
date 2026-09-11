from __future__ import annotations

from pathlib import Path

import numpy as np


class CameraError(RuntimeError):
    pass


class OpenCVCamera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV Python bindings are required for live camera use") from exc

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise CameraError(f"Could not open camera index {index}")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            raise CameraError("Camera did not return a frame")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        return rgb, bgr

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "OpenCVCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_rgb_image(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))

