"""Read-only final-path validation for Phase 3E's staged YOLO dataset."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from phase3d_prepare_training_data import EXPECTED_CASES, PILOT, PROJECT, digest, load_adjudications, validate_stage
from phase3b_validate_annotations import parse_yolo_line

STAGE = PROJECT / "datasets/derived_swarm_training_v2"
TRAIN_SOURCE = (PROJECT / "datasets/UAVSwarm-dataset-master/UAVSwarm-dataset-master/train").resolve()


def run() -> dict:
    errors: list[str] = []
    if not STAGE.is_dir():
        raise FileNotFoundError(STAGE)
    manifest_path = STAGE / "manifest.json"
    data_path = STAGE / "data.yaml"
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Staged manifest or data.yaml is missing")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = data.get("images", [])
    result = validate_stage(STAGE, records)
    errors.extend(result["errors"])
    expected_yaml = f"path: {STAGE.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: drone\n"
    if data_path.read_text(encoding="utf-8") != expected_yaml:
        errors.append("data.yaml differs from the intended final staging path/splits/class")
    if data.get("annotation_version") != "visible-airborne-v1.0-adjudicated":
        errors.append("manifest annotation version is unexpected")
    if data.get("dataset_a_used") is not False or data.get("dataset_b_supplied_annotations_used") is not False:
        errors.append("manifest does not explicitly exclude Dataset A and supplied Dataset B annotations")
    outcomes = data.get("adjudication_decisions", [])
    if {x.get("case_id") for x in outcomes} != EXPECTED_CASES:
        errors.append("staged manifest adjudication decision list does not match the flagged case set")
    if any(x.get("decision") not in ("keep", "remove") for x in outcomes):
        errors.append("staged manifest includes unresolved or unsupported adjudication")
    events, latest = load_adjudications()
    if len(latest) != len(EXPECTED_CASES):
        errors.append(f"latest case decision count {len(latest)} != {len(EXPECTED_CASES)}")
    event_by_case = {case_id: ev for case_id, ev in latest.items()}
    source_manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    source_by_id = {x["image_id"]: x for x in source_manifest["images"]}
    split_sequences = {"train": set(), "val": set()}
    boxes = Counter()
    for row in records:
        image_id = row.get("image_id")
        source = (PROJECT / row.get("source_image", "")).resolve()
        label_source = PROJECT / row.get("source_label", "")
        annotation_source = PROJECT / row.get("source_annotation", "")
        try:
            source.relative_to(TRAIN_SOURCE)
        except ValueError:
            errors.append(f"{image_id}: source path is not an official Dataset B training image")
        if source_by_id.get(image_id, {}).get("split") != row.get("split") or source_by_id.get(image_id, {}).get("sequence_id") != row.get("sequence_id"):
            errors.append(f"{image_id}: staged source split/sequence disagrees with pilot manifest")
        if not source.is_file() or not label_source.is_file() or not annotation_source.is_file():
            errors.append(f"{image_id}: source image/label/annotation missing")
            continue
        if digest(source) != row.get("image_sha256"):
            errors.append(f"{image_id}: image hash does not match original source")
        if digest(label_source) != row.get("label_sha256"):
            errors.append(f"{image_id}: label hash does not match final pilot label")
        annotation = json.loads(annotation_source.read_text(encoding="utf-8"))
        if annotation.get("status") != "reviewed":
            errors.append(f"{image_id}: source annotation not reviewed")
        yolo_rows = (STAGE / row["staged_label"]).read_text(encoding="utf-8").splitlines()
        json_boxes = annotation.get("boxes", [])
        if len(yolo_rows) != len(json_boxes):
            errors.append(f"{image_id}: staged YOLO rows and annotation boxes differ")
        w, h = row["width"], row["height"]
        for i, box in enumerate(json_boxes):
            try:
                coords = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
                if box.get("class_id", 0) != 0 or not all(math.isfinite(v) for v in coords):
                    raise ValueError("invalid class or nonfinite coordinate")
                x1, y1, x2, y2 = coords
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError("nonpositive or out-of-bounds dimensions")
                if i < len(yolo_rows):
                    _, cx, cy, bw, bh = parse_yolo_line(yolo_rows[i])
                    decoded = ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)
                    if max(abs(a-b) for a,b in zip(coords,decoded)) > 0.51:
                        errors.append(f"{image_id} box {i+1}: normalized staged coordinates disagree with JSON")
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{image_id} box {i+1}: {exc}")
        boxes[row["split"]] += len(json_boxes)
        split_sequences[row["split"]].add(row["sequence_id"])
        for event in event_by_case.values():
            if event["image_id"] == image_id and event["decision"] == "remove":
                if any(abs(float(b[k])-float(event["previous_coordinates"][k])) < 0.00011 for b in json_boxes for k in ("x1",)):
                    # Full coordinate equality is checked below; this early test only narrows without masking a mismatch.
                    if any(all(abs(float(b[k])-float(event["previous_coordinates"][k])) <= 0.00011 for k in ("x1","y1","x2","y2")) for b in json_boxes):
                        errors.append(f"{event['case_id']}: removed box still present in final annotation")
    overlap = sorted(split_sequences["train"] & split_sequences["val"])
    if overlap:
        errors.append(f"train/validation sequence overlap: {overlap}")
    if len(records) != 120 or len([r for r in records if r["split"] == "train"]) != 90 or len([r for r in records if r["split"] == "val"]) != 30:
        errors.append("staged dataset does not contain expected 90/30 manifest split")
    return {"staging_root": str(STAGE.resolve()), "data_yaml": str(data_path.resolve()), "images": len(records), "images_by_split": {k: sum(r["split"] == k for r in records) for k in ("train", "val")}, "boxes_by_split": dict(boxes), "total_boxes": sum(boxes.values()), "sequence_overlap": overlap, "unresolved_adjudications": sorted(k for k,v in latest.items() if v["decision"] not in ("keep","remove")), "official_test_excluded": all("/train/" in r["source_image"].replace("\\","/") for r in records), "supplied_annotations_used": data.get("dataset_b_supplied_annotations_used"), "dataset_a_used": data.get("dataset_a_used"), "errors": errors, "valid": not errors}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(2)
