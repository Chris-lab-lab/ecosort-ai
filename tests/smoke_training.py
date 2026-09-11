"""Explicit end-to-end pipeline check, not a trained waste-recognition model.

Run with the project Python: python tests/smoke_training.py
Creates synthetic images and a one-epoch model in a temporary directory, checks
INT8 export and the runtime wrapper, then removes the synthetic artifacts.
ImageNet backbone weights may be downloaded into the project's .cache folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("KERAS_HOME", str(PROJECT / ".cache" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> None:
    from ecosort_ai.model import WasteClassifier

    rng = np.random.default_rng(12)
    labels = ("general", "metal", "other", "plastic")
    colors = np.array(((200, 90, 20), (60, 170, 180), (40, 40, 50), (50, 150, 60)))
    with tempfile.TemporaryDirectory(prefix="ecosort-pipeline-smoke-") as directory:
        root = Path(directory)
        for split, count in (("train", 6), ("validation", 2)):
            for label, color in zip(labels, colors, strict=True):
                folder = root / split / label
                folder.mkdir(parents=True)
                for index in range(count):
                    pixels = np.clip(
                        color + rng.integers(-25, 25, size=(128, 128, 3)), 0, 255
                    ).astype(np.uint8)
                    Image.fromarray(pixels).save(folder / f"synthetic_{index}.png")

        result = subprocess.run(
            [
                sys.executable, str(PROJECT / "train.py"),
                "--data", str(root / "train"),
                "--validation-data", str(root / "validation"),
                "--output", str(root / "artifacts"),
                "--epochs", "1", "--fine-tune-epochs", "0", "--batch-size", "4",
            ],
            cwd=PROJECT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode:
            print(result.stdout[-12000:])
            print(result.stderr[-12000:], file=sys.stderr)
            raise SystemExit(result.returncode)

        artifacts = root / "artifacts"
        summary = json.loads((artifacts / "training_summary.json").read_text())
        classifier = WasteClassifier(
            artifacts / "waste_classifier_int8.tflite",
            artifacts / "labels.txt",
            delegate=None,
        )
        prediction = classifier.predict_rgb(np.full((128, 128, 3), 100, dtype=np.uint8))
        material_labels = set(labels) - {"other"}
        assert set(prediction.scores) == material_labels
        assert all(np.isfinite(value) for value in prediction.scores.values())
        assert prediction.supported_probability is not None
        assert prediction.prototype_distances is not None
        assert classifier.supports_unknown_rejection
        assert classifier.has_validity_output
        assert classifier.input["dtype"] == np.uint8
        assert classifier.output["dtype"] == np.uint8
        assert summary["model_type"] == "ecosort_v2_open_set"
        assert (artifacts / "open_set.json").is_file()
        matrix = summary["quantized_confusion_matrix"]["rows_are_actual_columns_are_predicted"]
        assert sum(sum(row) for row in matrix) == 8
        print("PASS: train -> best checkpoint -> full INT8 export -> validation -> runtime inference")
        print("Synthetic software test only. No waste accuracy claim; temporary model is discarded.")


if __name__ == "__main__":
    main()
