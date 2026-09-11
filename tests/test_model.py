from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ecosort_ai.model import WasteClassifier, read_labels


class ModelUtilitiesTests(unittest.TestCase):
    def test_uint8_quantization_round_trip(self) -> None:
        detail = {"dtype": np.dtype(np.uint8), "quantization": (1 / 255, 0)}
        values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        quantized = WasteClassifier._quantize(values, detail)
        restored = WasteClassifier._dequantize(quantized, detail)

        np.testing.assert_allclose(restored, values, atol=1 / 255)

    def test_duplicate_labels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.txt"
            path.write_text("plastic\nplastic\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_labels(path)


if __name__ == "__main__":
    unittest.main()
