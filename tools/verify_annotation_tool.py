"""Run isolated API/persistence/YOLO checks for annotation_server.py.

Uses one manifest-listed pilot image in a temporary subfolder under the new
derived-label directory, then removes only that test subfolder. It does not
change the real pilot annotation states or any source files.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

import annotation_server as server


def main() -> None:
    report_path = server.DATA_ROOT / "verification_report.json"
    test_root = server.DATA_ROOT / "_verification"
    if test_root.exists():
        raise SystemExit(f"Refusing to overwrite existing verification directory: {test_root}")
    server.ensure_data()
    source_manifest = server.load_manifest()
    sample = dict(source_manifest["images"][0])
    original_root = server.DATA_ROOT
    results: dict = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": {}, "errors": []}
    test_root.mkdir(parents=True)
    try:
        server.DATA_ROOT = test_root
        server.atomic_json(server.manifest_path(), {"schema_version": 1, "selection": {"train_sequences": [sample["sequence_id"]], "validation_sequences": []}, "images": [sample]})
        server.ensure_data()
        with TestClient(server.app) as client:
            results["checks"]["home_loads"] = client.get("/").status_code == 200
            results["checks"]["manifest_one_sample"] = len(client.get("/api/manifest").json()["images"]) == 1
            image_response = client.get(f"/api/image/{sample['image_id']}")
            results["checks"]["source_image_loads"] = image_response.status_code == 200 and image_response.headers.get("content-type", "").startswith("image/") and image_response.content[:2] == b"\xff\xd8"
            initial = client.get(f"/api/annotation/{sample['image_id']}").json()
            results["checks"]["starts_unreviewed"] = initial["status"] == "unreviewed" and initial["boxes"] == []
            test_box = {"x1": 10, "y1": 20, "x2": 110, "y2": 90}
            draft = client.post(f"/api/annotation/{sample['image_id']}", json={"status": "draft", "boxes": [test_box]})
            results["checks"]["draft_persists"] = draft.status_code == 200 and client.get(f"/api/annotation/{sample['image_id']}").json()["boxes"][0]["x2"] == 110
            reviewed = client.post(f"/api/annotation/{sample['image_id']}", json={"status": "reviewed", "boxes": [test_box]})
            label = server.label_path(sample).read_text(encoding="utf-8").split()
            normalized = [float(x) for x in label[1:]]
            cx, cy, bw, bh = normalized
            results["checks"]["reviewed_writes_yolo"] = reviewed.status_code == 200 and label[0] == "0" and len(normalized) == 4 and all(math.isfinite(x) for x in normalized)
            results["checks"]["normalized_coordinates_valid"] = 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1 and abs(cx - ((10+110)/2/sample["width"])) < 1e-7 and abs(cy - ((20+90)/2/sample["height"])) < 1e-7 and abs(bw - (100/sample["width"])) < 1e-7 and abs(bh - (70/sample["height"])) < 1e-7
            invalid = client.post(f"/api/annotation/{sample['image_id']}", json={"status": "reviewed", "boxes": [{"x1": -1, "y1": 1, "x2": 4, "y2": 5}]})
            results["checks"]["invalid_box_rejected"] = invalid.status_code == 422 and "Nothing was clipped or discarded" in invalid.text
            restored = client.post(f"/api/annotation/{sample['image_id']}", json={"status": "draft", "boxes": []})
            results["checks"]["draft_removes_training_label"] = restored.status_code == 200 and not server.label_path(sample).exists()
            validation = client.get("/api/progress").json()
            results["checks"]["progress_counts_reviewed_only"] = validation["reviewed"] == 0
        vreport = server.write_validation_report()
        results["checks"]["validation_report_valid"] = vreport["valid"] and vreport["train_validation_sequence_overlap"] == []
    except Exception as exc:
        results["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        server.DATA_ROOT = original_root
        shutil.rmtree(test_root)
    results["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results["passed"] = all(results["checks"].values()) and not results["errors"]
    server.atomic_json(report_path, results)
    print(json.dumps(results, indent=2))
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
