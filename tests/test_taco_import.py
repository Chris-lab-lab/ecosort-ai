from __future__ import annotations

import unittest

from scripts.import_taco import CLASS_MAP, _background_box, _safe_relative_file_name


class TacoImportTests(unittest.TestCase):
    def test_reviewed_taxonomy_mapping_has_all_sixty_categories(self) -> None:
        self.assertEqual(len(CLASS_MAP), 60)
        self.assertEqual(set(CLASS_MAP.values()), {"general", "metal", "other", "plastic", None})
        self.assertIsNone(CLASS_MAP["Unlabeled litter"])
        self.assertEqual(CLASS_MAP["Glass bottle"], "other")
        self.assertEqual(CLASS_MAP["Drink can"], "metal")

    def test_annotation_file_name_cannot_escape_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            _safe_relative_file_name("../outside.jpg")
        self.assertEqual(
            _safe_relative_file_name("batch_1/000001.jpg").as_posix(),
            "batch_1/000001.jpg",
        )

    def test_background_crop_avoids_annotated_object(self) -> None:
        crop = _background_box((640, 480), [(0.0, 0.0, 200.0, 200.0)])
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertGreaterEqual(crop[0], 400)


if __name__ == "__main__":
    unittest.main()
