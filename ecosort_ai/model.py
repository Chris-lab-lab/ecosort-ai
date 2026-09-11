from __future__ import annotations

import json
import os
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Prediction:
    scores: dict[str, float]
    supported_probability: float | None = None
    prototype_distances: dict[str, float] | None = None
    prototype_thresholds: dict[str, float] | None = None

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


def _tensor_width(detail: dict[str, Any]) -> int:
    shape = np.asarray(detail["shape"], dtype=np.int64)
    return int(np.prod(shape[1:]))


def _read_open_set_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"Open-set metadata file does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read open-set metadata {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Open-set metadata must contain a JSON object")
    return document


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
        self.outputs = list(self.interpreter.get_output_details())
        self.labels = read_labels(labels_path)

        if not self.outputs:
            raise ValueError("Model has no output tensors")
        material_outputs = [detail for detail in self.outputs if _tensor_width(detail) == len(self.labels)]
        if len(material_outputs) != 1:
            widths = [_tensor_width(detail) for detail in self.outputs]
            raise ValueError(
                f"Could not identify one {len(self.labels)}-class material output; "
                f"model output widths are {widths}"
            )
        # ``output`` remains the material tensor for compatibility with older callers.
        self.output = material_outputs[0]
        remaining = [detail for detail in self.outputs if detail is not self.output]
        validity_outputs = [detail for detail in remaining if _tensor_width(detail) == 1]
        if len(validity_outputs) > 1:
            raise ValueError("Model contains more than one scalar validity output")
        self.validity_output = validity_outputs[0] if validity_outputs else None
        embeddings = [detail for detail in remaining if detail is not self.validity_output]
        if len(embeddings) > 1:
            raise ValueError("Model contains more than one unrecognized feature output")
        self.embedding_output = embeddings[0] if embeddings else None

        explicit_metadata = Path(metadata_path) if metadata_path is not None else None
        automatic_metadata = Path(model_path).with_name("open_set.json")
        selected_metadata = explicit_metadata or (automatic_metadata if automatic_metadata.is_file() else None)
        self.metadata_path = selected_metadata
        self.open_set_metadata = _read_open_set_metadata(selected_metadata)
        metadata_labels = self.open_set_metadata.get("material_labels")
        if metadata_labels is not None:
            normalized_metadata_labels = [str(label).strip().lower() for label in metadata_labels]
            if normalized_metadata_labels != self.labels:
                raise ValueError(
                    "material_labels in open_set.json must exactly match labels.txt order"
                )
        threshold = float(self.open_set_metadata.get("validity_threshold", 0.5))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("validity_threshold in open-set metadata must be between 0 and 1")
        self.recommended_validity_threshold = threshold

        raw_prototypes = self.open_set_metadata.get("prototypes", {})
        raw_thresholds = self.open_set_metadata.get("prototype_thresholds", {})
        if not isinstance(raw_prototypes, dict) or not isinstance(raw_thresholds, dict):
            raise ValueError("prototypes and prototype_thresholds must be JSON objects")
        self.prototypes: dict[str, np.ndarray] = {}
        self.prototype_thresholds: dict[str, float] = {}
        embedding_width = _tensor_width(self.embedding_output) if self.embedding_output else None
        for label, raw_vector in raw_prototypes.items():
            normalized_label = str(label).strip().lower()
            if normalized_label not in self.labels:
                raise ValueError(f"Prototype label {normalized_label!r} is not in labels.txt")
            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
            if embedding_width is None or len(vector) != embedding_width:
                raise ValueError(
                    f"Prototype for {normalized_label} has {len(vector)} values; "
                    f"expected {embedding_width}"
                )
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError(f"Prototype for {normalized_label} is invalid")
            if normalized_label not in raw_thresholds:
                raise ValueError(f"Prototype threshold for {normalized_label} is missing")
            distance_threshold = float(raw_thresholds[normalized_label])
            if not math.isfinite(distance_threshold) or distance_threshold < 0.0:
                raise ValueError(f"Prototype threshold for {normalized_label} is invalid")
            self.prototypes[normalized_label] = vector / norm
            self.prototype_thresholds[normalized_label] = distance_threshold

        if bool(self.prototypes) != bool(self.prototype_thresholds):
            raise ValueError("Prototype vectors and thresholds must either both be present or both be absent")
        if self.prototypes and set(self.prototypes) != set(self.labels):
            raise ValueError("Open-set metadata needs one prototype for every material label")
        if self.prototypes and self.embedding_output is None:
            raise ValueError("Open-set metadata requires a model embedding output")
        if self.validity_output is not None and self.embedding_output is not None and not self.open_set_metadata:
            raise ValueError(
                "This open-set model requires open_set.json beside the model or --metadata PATH"
            )

        self.has_validity_output = self.validity_output is not None
        self.supports_unknown_rejection = "other" in self.labels or self.has_validity_output

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
        score_map = dict(zip(self.labels, (float(value) for value in scores), strict=True))

        supported_probability: float | None = None
        if self.validity_output is not None:
            raw_validity = self.interpreter.get_tensor(self.validity_output["index"])[0]
            decoded = self._dequantize(raw_validity, self.validity_output).reshape(-1)
            supported_probability = float(np.clip(decoded[0], 0.0, 1.0))

        distances: dict[str, float] | None = None
        if self.embedding_output is not None and self.prototypes:
            raw_embedding = self.interpreter.get_tensor(self.embedding_output["index"])[0]
            embedding = self._dequantize(raw_embedding, self.embedding_output).reshape(-1)
            norm = float(np.linalg.norm(embedding))
            if math.isfinite(norm) and norm > 0.0:
                embedding = embedding / norm
                distances = {
                    label: max(0.0, 1.0 - float(np.dot(embedding, prototype)))
                    for label, prototype in self.prototypes.items()
                }

        return Prediction(
            score_map,
            supported_probability=supported_probability,
            prototype_distances=distances,
            prototype_thresholds=(dict(self.prototype_thresholds) if distances is not None else None),
        )

    def describe(self) -> str:
        return json.dumps(
            {
                "input_shape": self.input["shape"].tolist(),
                "input_dtype": str(self.input["dtype"]),
                "outputs": [
                    {
                        "shape": detail["shape"].tolist(),
                        "dtype": str(detail["dtype"]),
                    }
                    for detail in self.outputs
                ],
                "labels": self.labels,
                "model_type": "dual_head_open_set" if self.has_validity_output else "legacy_softmax",
                "validity_threshold": (
                    self.recommended_validity_threshold if self.has_validity_output else None
                ),
                "prototype_rejection": bool(self.prototypes),
                "metadata": str(self.metadata_path) if self.metadata_path else None,
                "runtime": self.runtime_name,
                "delegate": self.delegate_path or "CPU",
            },
            indent=2,
        )
