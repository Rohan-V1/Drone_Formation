import json
import tempfile
import unittest
from pathlib import Path

from tools.phase3b_validate_annotations import audit_pilot, parse_yolo_line, read_annotation


class AnnotationReaderTests(unittest.TestCase):
    def test_absent_is_unreviewed_not_reviewed_empty(self):
        with tempfile.TemporaryDirectory() as d:
            result = read_annotation(Path(d) / "missing.json", "sample")
        self.assertEqual(result["status"], "unreviewed")
        self.assertEqual(result["boxes"], [])

    def test_reviewed_empty_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.json"
            path.write_text(json.dumps({"image_id": "sample", "status": "reviewed", "boxes": []}), encoding="utf-8")
            result = read_annotation(path, "sample")
        self.assertEqual(result["status"], "reviewed")
        self.assertEqual(result["boxes"], [])


class YoloParserTests(unittest.TestCase):
    def test_valid_label(self):
        self.assertEqual(parse_yolo_line("0 0.5 0.5 0.2 0.4"), (0, 0.5, 0.5, 0.2, 0.4))

    def test_rejects_invalid_lines(self):
        for line in ("1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 0 0.5 0.3 0.2", "0 0.5 0.5 0.2"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_yolo_line(line)


class ValidatorTests(unittest.TestCase):
    def test_real_pilot_has_valid_correspondence_and_split(self):
        from tools.phase3b_validate_annotations import discover_pilot
        result = audit_pilot(discover_pilot())
        self.assertTrue(result["structural_valid"], "\n".join(result["errors"]))
        self.assertEqual(result["counts"]["manifest_images"], 120)
        self.assertEqual(result["sequence_isolation"]["overlap"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
