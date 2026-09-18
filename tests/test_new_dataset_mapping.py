from __future__ import annotations

import unittest

from scripts.map_new_dataset_manifest import mapped_label


class NewDatasetMappingTests(unittest.TestCase):
    def test_approved_subclasses(self) -> None:
        expected = {
            "Recyclable - Others/Glass/a.jpg": "other",
            "Recyclable - Others/Metals/a.jpg": "metal",
            "Recyclable - Others/Plastics/a.jpg": "plastic",
            "Recyclable - Paper/Cardboard/a.jpg": "general",
            "Recyclable - Paper/Paper/a.jpg": "general",
            "Regular Trash/Tissue/a.jpg": "general",
            "Regular Trash/Plastic Bags/a.jpg": "general",
        }
        for source, label in expected.items():
            with self.subTest(source=source):
                self.assertEqual(mapped_label(source, "general"), label)

    def test_unknown_subclass_is_not_guessed(self) -> None:
        with self.assertRaisesRegex(ValueError, "No approved"):
            mapped_label("Regular Trash/Batteries/a.jpg", "general")


if __name__ == "__main__":
    unittest.main()
