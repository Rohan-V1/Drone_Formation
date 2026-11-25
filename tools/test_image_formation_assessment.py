"""Unit tests for the still-image formation gate using Phase 5A rules."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai-service"))
from src.app import _assess_image_formation, _suppress_duplicate_image_boxes

PHASE5A = ROOT / "outputs" / "phase5a_formation_analysis" / "run_004" / "formation_classifier.py"
spec = importlib.util.spec_from_file_location("phase5a_image_geometry_test", PHASE5A)
assert spec is not None and spec.loader is not None
geometry_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry_module
spec.loader.exec_module(geometry_module)


def image_case(points):
    points = points * 0.08 + 0.5
    centers = points * 1000
    boxes = [(x - 5, y - 5, x + 5, y + 5) for x, y in centers]
    return points.tolist(), boxes


class ImageFormationAssessmentTests(unittest.TestCase):
    def test_supported_static_geometries_use_phase5a_templates(self):
        templates = geometry_module.canonical_templates()
        for template, expected in (("V formation", "V"), ("Diamond/Box formation", "Diamond/Box"), ("Circle formation", "Circle")):
            with self.subTest(template=template):
                points, boxes = image_case(templates[template])
                result = _assess_image_formation(points, boxes, geometry_module.classify)
                self.assertEqual(result["label"], expected)
                self.assertFalse(result["overlapping_detection_pairs"])

    def test_collinear_and_sparse_geometry_stay_unknown(self):
        templates = geometry_module.canonical_templates()
        points, boxes = image_case(templates["Column/File formation"])
        collinear = _assess_image_formation(points, boxes, geometry_module.classify)
        self.assertEqual(collinear["label"], "Unknown")
        self.assertIn("collinear", collinear["reason"])
        sparse_points, sparse_boxes = image_case(templates["V formation"][:3])
        sparse = _assess_image_formation(sparse_points, sparse_boxes, geometry_module.classify)
        self.assertEqual(sparse["label"], "Unknown")

    def test_overlapping_boxes_reject_spatial_assessment(self):
        points, boxes = image_case(geometry_module.canonical_templates()["V formation"])
        boxes[1] = boxes[0]
        result = _assess_image_formation(points, boxes, geometry_module.classify)
        self.assertEqual(result["label"], "Unknown")
        self.assertTrue(result["overlapping_detection_pairs"])

    def test_geometry_is_rotation_translation_and_scale_invariant(self):
        points = geometry_module.canonical_templates()["V formation"]
        angle = 0.63
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        transformed = points @ rotation.T * 2.7 + np.array([8.0, -4.0])
        classified = geometry_module.classify(transformed)
        self.assertEqual(classified.label, "V formation")

    def test_duplicate_cleanup_requires_overlap_containment_and_close_centers(self):
        boxes = [(100, 100, 140, 130), (103, 101, 142, 132), (146, 100, 156, 110)]
        kept, suppressed = _suppress_duplicate_image_boxes(boxes, [0.60, 0.40, 0.35])
        self.assertEqual(kept, [0, 2])
        self.assertEqual([item["detection_id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["representative_detection_id"], 1)
        self.assertGreaterEqual(suppressed[0]["iou"], 0.50)
        self.assertGreaterEqual(suppressed[0]["intersection_over_smaller"], 0.85)
        self.assertLessEqual(suppressed[0]["center_distance_smaller_diagonal_ratio"], 0.20)

    def test_close_neighboring_boxes_are_not_suppressed_without_strong_overlap(self):
        boxes = [(0, 0, 10, 10), (6, 0, 16, 10)]
        kept, suppressed = _suppress_duplicate_image_boxes(boxes, [0.60, 0.55])
        self.assertEqual(kept, [0, 1])
        self.assertFalse(suppressed)

    def test_shared_multidrone_example_is_v_after_measured_duplicates_removed(self):
        artifact = ROOT / "outputs" / "real_image_jobs" / "92efa3e7-2fdd-4011-bafa-9496e3976623" / "metadata.json"
        if not artifact.is_file():
            self.skipTest("Saved shared-image job artifact is not available")
        metadata = json.loads(artifact.read_text(encoding="utf-8"))
        raw = metadata["detections"]
        boxes = [(row["x1"], row["y1"], row["x2"], row["y2"]) for row in raw]
        confidences = [row["confidence"] for row in raw]
        kept, suppressed = _suppress_duplicate_image_boxes(boxes, confidences)
        self.assertEqual(len(raw), 8)
        self.assertEqual([(row["detection_id"], row["representative_detection_id"]) for row in suppressed], [(5, 2), (7, 4)])
        self.assertEqual(len(kept), 6)
        width, height = metadata["width"], metadata["height"]
        clean_boxes = [boxes[i] for i in kept]
        points = [((b[0] + b[2]) / (2 * width), (b[1] + b[3]) / (2 * height)) for b in clean_boxes]
        result = _assess_image_formation(points, clean_boxes, geometry_module.classify)
        self.assertEqual(result["label"], "V")
        self.assertGreater(result["quality"], 0.80)
        self.assertFalse(result["overlapping_detection_pairs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
