"""Runtime wrapper for the experimental segmentation + material TFLite model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model import _runtime, find_ethosu_delegate, read_labels


@dataclass(frozen=True)
class FullPrediction:
    """One prediction from the compact multi-task model."""

    scores: dict[str, float]
    supported_probability: float
    object_mask: np.ndarray

    @property
    def top(self) -> tuple[str, float]:
        return max(self.scores.items(), key=lambda item: item[1])


def _tensor_width(detail: dict[str, Any]) -> int:
    shape = np.asarray(detail["shape"], dtype=np.int64)
    return int(np.prod(shape[1:]))


class FullWasteModel:
    """Run material, validity, and foreground-mask heads with one interpreter."""

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        *,
        delegate: str | None = "auto",
        threads: int = 2,
        metadata_path: str | Path | None = None,
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
                raise RuntimeError(
                    f"Could not load TensorFlow Lite delegate {delegate}: {exc}"
                ) from exc
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
        outputs = list(self.interpreter.get_output_details())
        self.labels = read_labels(labels_path)

        material = [
            detail
            for detail in outputs
            if len(detail["shape"]) <= 2
            and _tensor_width(detail) == len(self.labels)
        ]
        validity = [
            detail
            for detail in outputs
            if len(detail["shape"]) <= 2 and _tensor_width(detail) == 1
        ]
        masks = [
            detail
            for detail in outputs
            if len(detail["shape"]) == 4
            and int(detail["shape"][-1]) == 1
            and int(detail["shape"][1]) > 1
            and int(detail["shape"][2]) > 1
        ]
        if len(material) != 1 or len(validity) != 1 or len(masks) != 1:
            shapes = [list(map(int, detail["shape"])) for detail in outputs]
            raise ValueError(
                "Expected exactly one material, validity, and object-mask output; "
                f"received output shapes {shapes}"
            )
        self.material_output = material[0]
        self.validity_output = validity[0]
        self.mask_output = masks[0]

        shape = self.input["shape"]
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[3]) != 3:
            raise ValueError(f"Expected NHWC RGB model input; received {shape}")
        self.height = int(shape[1])
        self.width = int(shape[2])
        self.mask_height = int(self.mask_output["shape"][1])
        self.mask_width = int(self.mask_output["shape"][2])

        selected_metadata = (
            Path(metadata_path)
            if metadata_path is not None
            else Path(model_path).with_name("full_model.json")
        )
        self.metadata: dict[str, Any] = {}
        if selected_metadata.is_file():
            self.metadata = json.loads(
                selected_metadata.read_text(encoding="utf-8")
            )
        self.validity_threshold = float(
            self.metadata.get("validity_threshold", 0.5)
        )
        self.material_threshold = float(
            self.metadata.get("material_threshold", 0.70)
        )

    @staticmethod
    def _quantize(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
        dtype = detail["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero = detail["quantization"]
            if not scale:
                raise ValueError("Integer tensor has no quantization scale")
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

    def predict_rgb(self, rgb_image: np.ndarray) -> FullPrediction:
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image")
        try:
            import cv2

            resized = cv2.resize(
                rgb_image, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        except ImportError:
            from PIL import Image

            resized = np.asarray(
                Image.fromarray(rgb_image).resize(
                    (self.width, self.height), Image.Resampling.BILINEAR
                )
            )

        tensor = self._quantize(resized.astype(np.float32), self.input)[None, ...]
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()

        material = self._dequantize(
            self.interpreter.get_tensor(self.material_output["index"])[0],
            self.material_output,
        ).reshape(-1)
        validity = self._dequantize(
            self.interpreter.get_tensor(self.validity_output["index"])[0],
            self.validity_output,
        ).reshape(-1)
        mask = self._dequantize(
            self.interpreter.get_tensor(self.mask_output["index"])[0],
            self.mask_output,
        )
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = np.clip(mask, 0.0, 1.0)
        return FullPrediction(
            scores=dict(
                zip(self.labels, (float(value) for value in material), strict=True)
            ),
            supported_probability=float(validity[0]),
            object_mask=mask,
        )

    def safe_to_route(self, prediction: FullPrediction) -> bool:
        if not self.metadata.get("validity_trained", True):
            return False
        _, confidence = prediction.top
        return (
            prediction.supported_probability >= self.validity_threshold
            and confidence >= self.material_threshold
        )

    def describe(self) -> str:
        return json.dumps(
            {
                "runtime": self.runtime_name,
                "delegate": self.delegate_path or "CPU",
                "input": [self.height, self.width, 3],
                "mask": [self.mask_height, self.mask_width],
                "labels": self.labels,
                "taxonomy": self.metadata.get("taxonomy", "material"),
                "validity_trained": self.metadata.get("validity_trained", True),
                "validity_threshold": self.validity_threshold,
                "material_threshold": self.material_threshold,
            },
            indent=2,
        )
