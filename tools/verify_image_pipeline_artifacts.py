"""Validate downloaded detector-only image job artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(root / "annotated.png"), cv2.IMREAD_COLOR)
    assert image is not None, "annotated.png did not decode"
    height, width = image.shape[:2]
    assert width == metadata["width"] and height == metadata["height"], "image dimensions differ from metadata"
    with (root / "detections.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "class_id", "class_name", "track_id", "confidence", "x1", "y1", "x2", "y2"}
        assert required.issubset(reader.fieldnames or []), "detection CSV missing required columns"
        rows = list(reader)
    assert len(rows) == metadata["detection_count"], "CSV and metadata detection counts differ"
    raw_count = metadata.get("raw_detection_count", metadata["detection_count"])
    distinct_count = metadata.get("estimated_distinct_drone_count", raw_count)
    assert raw_count == len(rows), "raw detector count and CSV row count differ"
    cleaned = metadata.get("cleaned_detections", rows)
    assert distinct_count == len(cleaned), "estimated distinct count and cleaned detection list differ"
    if "counted_as_distinct" in (reader.fieldnames or []):
        counted_rows = sum(row["counted_as_distinct"].lower() == "true" for row in rows)
        assert counted_rows == distinct_count, "CSV distinct flags and estimated count differ"
    for row in rows:
        assert row["class_id"] == "0" and row["class_name"] == "drone"
        assert row["track_id"] == "", "single image should not invent a track ID"
        x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height, f"out-of-bounds box: {row}"
    allowed_formations = {"Unknown", "V", "Echelon", "Column/File", "Line/Abreast", "Diamond/Box", "Circle"}
    assert metadata["formation_prediction"] in allowed_formations
    if distinct_count < 4 or metadata.get("overlapping_detection_pairs"):
        assert metadata["formation_prediction"] == "Unknown", "sparse or overlapping detections must not produce a formation candidate"
    assert metadata.get("formation_method", "").startswith("Phase 5A conservative image-space geometry")
    print(json.dumps({"annotated_image_decoded": True, "dimensions": [width, height],
                      "raw_detection_rows": len(rows), "estimated_distinct_drones": distinct_count,
                      "suppressed_likely_duplicates": len(metadata.get("duplicate_suppressions", [])),
                      "boxes_in_bounds": True, "track_ids_invented": False,
                      "formation": metadata["formation_prediction"],
                      "formation_reason": metadata["formation_reason"],
                      "device": metadata["device"], "gpu": metadata["gpu_name"]}, indent=2))


if __name__ == "__main__":
    main()
