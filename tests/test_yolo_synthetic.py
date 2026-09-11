"""Small adversarial fixtures for the independently saved dataset validator."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from yolo_synthetic.validate_dataset import validate_dataset


class SyntheticDatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        (self.dataset / "data.yaml").write_text(
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: plastic_bottle\n  1: metal_can\n  2: wrapper\n",
            encoding="utf-8",
        )
        self.rows = []
        for split_index, split in enumerate(("train", "val", "test")):
            for negative in (False, True):
                name = "negative" if negative else "objects"
                image = np.full((32, 32, 3), 40 + split_index * 40 + int(negative) * 10, dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                objects, labels = [], []
                if not negative:
                    for class_id in range(3):
                        x1, y1, x2, y2 = 2 + class_id * 10, 4, 8 + class_id * 10, 20
                        mask[y1:y2, x1:x2] = class_id + 1
                        image[y1:y2, x1:x2, class_id] = 255
                        objects.append({"instance_id": class_id + 1, "class_id": class_id, "bbox_xyxy": [x1, y1, x2, y2]})
                        labels.append(f"{class_id} {(x1 + x2) / 64} {(y1 + y2) / 64} {(x2 - x1) / 32} {(y2 - y1) / 32}")
                paths = {"image": f"images/{split}/{name}.jpg", "label": f"labels/{split}/{name}.txt", "mask": f"masks/{split}/{name}.png"}
                for relative in paths.values():
                    (self.dataset / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(self.dataset / paths["image"])
                Image.fromarray(mask).save(self.dataset / paths["mask"])
                (self.dataset / paths["label"]).write_text("\n".join(labels), encoding="utf-8")
                self.rows.append({**paths, "split": split, "seed": split_index * 2 + int(negative), "width": 32, "height": 32, "objects": objects})
        self.write_manifest()

    def write_manifest(self):
        (self.dataset / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in self.rows) + "\n", encoding="utf-8")

    def test_valid_dataset_with_negatives(self):
        report = validate_dataset(self.dataset)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["images"], 6)
        self.assertEqual(report["objects"], 9)
        self.assertEqual(report["splits"]["test"]["negative_images"], 1)

    def test_rejects_loose_bbox_even_if_label_matches_manifest(self):
        self.rows[0]["objects"][0]["bbox_xyxy"] = [0, 0, 10, 22]
        self.write_manifest()
        label_path = self.dataset / self.rows[0]["label"]
        lines = label_path.read_text(encoding="utf-8").splitlines()
        lines[0] = "0 0.15625 0.34375 0.3125 0.6875"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        report = validate_dataset(self.dataset)
        self.assertFalse(report["valid"])
        self.assertTrue(any("not tight around visible mask" in error for error in report["errors"]))

    def test_rejects_duplicate_image_across_splits(self):
        shutil.copyfile(self.dataset / self.rows[0]["image"], self.dataset / self.rows[2]["image"])
        report = validate_dataset(self.dataset)
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicate image across splits" in error for error in report["errors"]))

    def test_rejects_nonfinite_label(self):
        label_path = self.dataset / self.rows[0]["label"]
        label_path.write_text("0 nan 0.375 0.1875 0.5\n1 0.46875 0.375 0.1875 0.5\n2 0.78125 0.375 0.1875 0.5\n", encoding="utf-8")
        report = validate_dataset(self.dataset)
        self.assertFalse(report["valid"])
        self.assertTrue(any("finite and normalized" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
