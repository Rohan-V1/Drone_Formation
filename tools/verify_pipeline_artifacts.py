"""Verify downloaded artifacts from tools/test_video_pipeline.ps1."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact_dir", type=Path)
    args = p.parse_args()
    root = args.artifact_dir
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["frames_processed"] > 0
    cap = cv2.VideoCapture(str(root / "annotated.mp4"))
    assert cap.isOpened(), "annotated.mp4 did not open"
    decoded = 0
    try:
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            decoded += 1
    finally:
        cap.release()
    assert decoded == metadata["frames_processed"], f"decoded {decoded}, expected {metadata['frames_processed']}"
    with (root / "detections.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame_number", "track_id", "confidence", "x1", "y1", "x2", "y2"}
        assert required.issubset(reader.fieldnames or []), "detection CSV missing required columns"
        detection_rows = list(reader)
    width, height = int(metadata["width"]), int(metadata["height"])
    for row in detection_rows:
        x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height, f"out-of-bounds box: {row}"
    with (root / "formations.csv").open(newline="", encoding="utf-8") as f:
        formations = list(csv.DictReader(f))
    assert formations, "no formation windows were written"
    assert len(formations) == len(metadata["windows"]), "window CSV and metadata counts differ"
    assert all(r["prediction"] == "Unknown" or r["prediction"] in {"V", "Echelon", "Column/File", "Line/Abreast", "Diamond/Box", "Circle"} for r in formations)
    print(json.dumps({"video_decoded_frames": decoded, "detection_rows": len(detection_rows),
                      "formation_windows": len(formations), "all_boxes_in_bounds": True,
                      "checkpoint": metadata["detector_checkpoint"], "device": metadata["device"],
                      "gpu": metadata["gpu_name"]}, indent=2))


if __name__ == "__main__":
    main()
