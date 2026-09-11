from __future__ import annotations

import json
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Prediction:
    scores: dict[str, float]

    @property
    def top(self) -> tuple[str, float]:
        return max(self.scores.items(), key=lambda item: item[1])


def _runtime() -> tuple[Any, Any, str]:
    """Prefer the small board runtime, then fall back to full TensorFlow."""
    for module_name in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter"):
        try:
            import importlib

            module = importlib.import_module(module_name)
            return module.Interpreter, module.load_delegate, module_name
        except ImportError:
            pass
    try:
        import tensorflow as tf

        return tf.lite.Interpreter, tf.lite.experimental.load_delegate, "tensorflow"
    except ImportError as exc:
        raise RuntimeError(
            "Install the BSP tflite_runtime/ai_edge_litert package on the board "
            "or tensorflow on the laptop."
        ) from exc


def read_labels(path: str | Path) -> list[str]:
    labels = [line.strip().lower() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate labels found in {path}")
    return labels


def find_ethosu_delegate() -> str | None:
    """Locate the delegate across common NXP BSP library layouts."""

    candidates = ["/usr/lib/libethosu_delegate.so"]
    candidates.extend(sorted(glob.glob("/usr/lib*/libethosu_delegate.so*")))
    return next((path for path in dict.fromkeys(candidates) if os.path.isfile(path)), None)


class WasteClassifier:
    """Small TensorFlow Lite wrapper that supports CPU or Ethos-U delegate."""

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        *,
        delegate: str | None = "auto",
        threads: int = 2,
    ) -> None:
        Interpreter, load_delegate, runtime_name = _runtime()
        delegates = []
        self.runtime_name = runtime_name
        self.delegate_path: str | None = None
        if delegate == "auto":
            delegate = find_ethosu_delegate()
        if delegate:
            try:
                delegates.append(load_delegate(delegate))
            except Exception as exc:
                raise RuntimeError(f"Could not load TensorFlow Lite delegate {delegate}: {exc}") from exc
            self.delegate_path = delegate

        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "num_threads": threads,
        }
        if delegates:
            kwargs["experimental_delegates"] = delegates
        self.interpreter = Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.labels = read_labels(labels_path)

        shape = self.input["shape"]
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[3]) != 3:
            raise ValueError(f"Expected NHWC RGB model input; received {shape}")
        self.height = int(shape[1])
        self.width = int(shape[2])

    @staticmethod
    def _quantize(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
        dtype = detail["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero = detail["quantization"]
            if not scale:
                raise ValueError("Integer input tensor has no quantization scale")
            limits = np.iinfo(dtype)
            array = np.rint(array / scale + zero)
            return np.clip(array, limits.min, limits.max).astype(dtype)
        return array.astype(dtype)

    @staticmethod
    def _dequantize(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
        if np.issubdtype(detail["dtype"], np.integer):
            scale, zero = detail["quantization"]
            return (array.astype(np.float32) - zero) * scale
        return array.astype(np.float32)

    def predict_rgb(self, rgb_image: np.ndarray) -> Prediction:
        """Predict from an HxWx3 uint8 RGB image."""
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image")

        # OpenCV is already used for the live camera. Pillow is the fallback.
        try:
            import cv2

            resized = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image

            resized = np.asarray(
                Image.fromarray(rgb_image).resize((self.width, self.height), Image.Resampling.BILINEAR)
            )

        real_input = resized.astype(np.float32)
        tensor = self._quantize(real_input, self.input)[None, ...]
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"])[0]
        scores = self._dequantize(raw, self.output).reshape(-1)

        if len(scores) != len(self.labels):
            raise ValueError(
                f"Model has {len(scores)} outputs but labels file has {len(self.labels)} entries"
            )
        return Prediction(dict(zip(self.labels, (float(value) for value in scores), strict=True)))

    def describe(self) -> str:
        return json.dumps(
            {
                "input_shape": self.input["shape"].tolist(),
                "input_dtype": str(self.input["dtype"]),
                "output_shape": self.output["shape"].tolist(),
                "output_dtype": str(self.output["dtype"]),
                "labels": self.labels,
                "runtime": self.runtime_name,
                "delegate": self.delegate_path or "CPU",
            },
            indent=2,
        )
