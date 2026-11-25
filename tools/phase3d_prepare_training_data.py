"""Create and validate an immutable copy of the adjudicated 120-image pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from phase3b_validate_annotations import PROJECT, audit_pilot, parse_yolo_line

PILOT = PROJECT / "datasets/derived_swarm_labels_v1"
DATASETS = PROJECT / "datasets"
DEFAULT_OUTPUT = DATASETS / "derived_swarm_training_v2"
EXPECTED_CASES = {
    "UAVSwarm-29_000042#1", "UAVSwarm-29_000042#2",
    "UAVSwarm-29_000059#1", "UAVSwarm-29_000059#2", "UAVSwarm-29_000059#3",
    "UAVSwarm-29_000092#6", "UAVSwarm-29_000092#7",
    "UAVSwarm-29_000125#2", "UAVSwarm-29_000125#3",
    "UAVSwarm-27_000010#1",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_adjudications() -> tuple[list[dict], dict[str, dict]]:
    log = PILOT / "annotation_change_log.jsonl"
    if not log.is_file():
        raise RuntimeError(f"Adjudication log is missing: {log}")
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    applications = {e.get("record_id"): e for e in events if e.get("event_type") == "application"}
    latest = {}
    for event in events:
        if event.get("event_type") == "adjudication" and event.get("case_id"):
            latest[event["case_id"]] = event
    unresolved = sorted(EXPECTED_CASES - set(latest))
    if unresolved:
        raise RuntimeError(f"Adjudication decisions are pending: {unresolved}")
    outcomes = {}
    for case_id, event in latest.items():
        application = applications.get(event.get("record_id"), {})
        if not application.get("applied"):
            raise RuntimeError(f"Adjudication application is incomplete for {case_id}")
        if event.get("decision") not in ("keep", "remove"):
            raise RuntimeError(f"Uncertain or unsupported decision for {case_id}: {event.get('decision')}")
        outcomes[case_id] = {**event, "application": application}
    if set(outcomes) != EXPECTED_CASES:
        raise RuntimeError(f"Unexpected adjudication case set: {sorted(set(outcomes) ^ EXPECTED_CASES)}")
    return events, outcomes


def validate_stage(stage: Path, records: list[dict]) -> dict:
    errors: list[str] = []
    counts = Counter(r["split"] for r in records)
    boxes = Counter()
    train_seq = {r["sequence_id"] for r in records if r["split"] == "train"}
    val_seq = {r["sequence_id"] for r in records if r["split"] == "val"}
    overlap = sorted(train_seq & val_seq)
    if overlap:
        errors.append(f"train/validation sequence overlap: {overlap}")
    if len(records) != 120 or counts != Counter({"train": 90, "val": 30}):
        errors.append(f"unexpected staged image counts: {dict(counts)} of {len(records)}")
    ids = [r["image_id"] for r in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate image IDs in staged manifest")
    for split in ("train", "val"):
        expected_stems = {r["image_id"] for r in records if r["split"] == split}
        image_files = {p.stem for p in (stage / "images" / split).glob("*") if p.is_file()}
        label_files = {p.stem for p in (stage / "labels" / split).glob("*.txt") if p.is_file()}
        if image_files != expected_stems:
            errors.append(f"{split}: staged image file correspondence mismatch; missing={sorted(expected_stems-image_files)}, extra={sorted(image_files-expected_stems)}")
        if label_files != expected_stems:
            errors.append(f"{split}: staged label file correspondence mismatch; missing={sorted(expected_stems-label_files)}, extra={sorted(label_files-expected_stems)}")
    for row in records:
        image = stage / row["staged_image"]
        label = stage / row["staged_label"]
        try:
            with Image.open(image) as im:
                im.verify()
            with Image.open(image) as im:
                dims = im.size
            if dims != (row["width"], row["height"]):
                errors.append(f"{row['image_id']}: staged dimensions {dims} differ from manifest")
            if digest(image) != row["image_sha256"]:
                errors.append(f"{row['image_id']}: staged image hash differs from source")
        except Exception as exc:
            errors.append(f"{row['image_id']}: staged image unreadable: {exc}")
        try:
            text = label.read_text(encoding="utf-8")
            if digest(label) != row["label_sha256"]:
                errors.append(f"{row['image_id']}: staged label hash differs from source label")
            parsed = [parse_yolo_line(line) for line in text.splitlines()]
            boxes[row["split"]] += len(parsed)
            if len(parsed) != row["box_count"]:
                errors.append(f"{row['image_id']}: staged rows {len(parsed)} vs manifest boxes {row['box_count']}")
        except Exception as exc:
            errors.append(f"{row['image_id']}: invalid staged label: {exc}")
    yaml_path = stage / "data.yaml"
    if not yaml_path.is_file():
        errors.append("data.yaml missing from final staging path")
    else:
        yaml = yaml_path.read_text(encoding="utf-8")
        for expected in ("train: images/train", "val: images/val", "0: drone"):
            if expected not in yaml:
                errors.append(f"data.yaml missing expected entry: {expected}")
    return {"staging_root": str(stage.resolve()), "image_count": len(records), "images_by_split": dict(counts), "boxes_by_split": dict(boxes), "total_boxes": sum(boxes.values()), "train_sequences": sorted(train_seq), "validation_sequences": sorted(val_seq), "sequence_overlap": overlap, "validation_from_final_staged_paths": True, "errors": errors, "valid": not errors}


def prepare(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {output}")
    weights = PROJECT / "yolov8n.pt"
    run_dir = PROJECT / "runs/phase3d_baseline"
    if not weights.is_file():
        raise RuntimeError(f"Local YOLOv8n starting weights missing: {weights}; refusing any download fallback")
    if run_dir.exists():
        raise FileExistsError(f"Proposed training output already exists; choose a new run directory: {run_dir}")
    structural = audit_pilot(PILOT)
    if not structural["structural_valid"]:
        raise RuntimeError("Source pilot fails structural validation: " + "; ".join(structural["errors"][:20]))
    events, outcomes = load_adjudications()
    source_manifest_path = PILOT / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_items = source_manifest["images"]
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    records = []
    train_root = (PROJECT / "datasets/UAVSwarm-dataset-master/UAVSwarm-dataset-master/train").resolve()
    for item in source_items:
        image_id, split = item["image_id"], item["split"]
        source_image = (PROJECT / item["source_path"]).resolve()
        if not source_image.is_relative_to(train_root):
            raise RuntimeError(f"Refusing non-training-sequence source image: {source_image}")
        source_label = PILOT / "labels" / split / f"{image_id}.txt"
        annotation = PILOT / "annotations" / split / f"{image_id}.json"
        source_json = json.loads(annotation.read_text(encoding="utf-8"))
        if source_json.get("status") != "reviewed":
            raise RuntimeError(f"{image_id}: source annotation is not reviewed")
        with Image.open(source_image) as im:
            im.verify()
        with Image.open(source_image) as im:
            width, height = im.size
        if (width, height) != (item["width"], item["height"]):
            raise RuntimeError(f"{image_id}: source dimensions disagree with pilot manifest")
        # Copy only source images and the pilot's own reviewed YOLO text labels.
        staged_image = output / "images" / split / f"{image_id}{source_image.suffix.lower()}"
        staged_label = output / "labels" / split / f"{image_id}.txt"
        shutil.copy2(source_image, staged_image)
        shutil.copy2(source_label, staged_label)
        rows = [line for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != len(source_json.get("boxes", [])):
            raise RuntimeError(f"{image_id}: source JSON/YOLO counts disagree while staging")
        # Validate every copied label before writing it into the dataset manifest.
        for line in rows:
            parse_yolo_line(line)
        records.append({
            "image_id": image_id,
            "source_image": item["source_path"],
            "source_annotation": str(annotation.relative_to(PROJECT).as_posix()),
            "source_label": str(source_label.relative_to(PROJECT).as_posix()),
            "sequence_id": item["sequence_id"],
            "frame_number": item["frame_number"],
            "split": split,
            "width": width,
            "height": height,
            "box_count": len(rows),
            "staged_image": f"images/{split}/{staged_image.name}",
            "staged_label": f"labels/{split}/{staged_label.name}",
            "image_sha256": digest(source_image),
            "label_sha256": digest(source_label),
        })
    yaml_text = f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: drone\n"
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8", newline="\n")
    log_path = PILOT / "annotation_change_log.jsonl"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_name": "Manually reviewed UAVSwarm airborne-drone pilot staging",
        "annotation_version": "visible-airborne-v1.0-adjudicated",
        "annotation_policy": str((PILOT / "ANNOTATION_POLICY.md").relative_to(PROJECT).as_posix()),
        "source_pilot_manifest": str(source_manifest_path.relative_to(PROJECT).as_posix()),
        "source_pilot_manifest_sha256": digest(source_manifest_path),
        "adjudication_log": str(log_path.relative_to(PROJECT).as_posix()),
        "adjudication_log_sha256": digest(log_path),
        "adjudication_decisions": [{"case_id": k, "decision": v["decision"], "reason": v["reason"], "record_id": v["record_id"], "timestamp_utc": v["timestamp_utc"]} for k, v in sorted(outcomes.items())],
        "dataset_b_supplied_annotations_used": False,
        "dataset_a_used": False,
        "class_names": ["drone"],
        "images": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = validate_stage(output, records)
    training_command = (
        f'& ".\\.venv311\\Scripts\\yolo.exe" detect train `\n'
        f'  model="{weights}" `\n'
        f'  data="{(output / "data.yaml")}" `\n'
        f'  imgsz=640 batch=8 epochs=100 patience=15 seed=42 deterministic=True device=0 workers=4 `\n'
        f'  project="{(PROJECT / "runs")}" name="phase3d_baseline" exist_ok=False\n'
    )
    (output / "TRAINING_COMMAND_DRAFT.ps1").write_text(
        "# Draft only. Recheck GPU memory and approval gates before running.\n" + training_command,
        encoding="utf-8", newline="\n",
    )
    result.update({"source_pilot_structural_valid": structural["structural_valid"], "adjudicated_cases": len(outcomes), "decisions": dict(Counter(v["decision"] for v in outcomes.values())), "annotation_version": manifest["annotation_version"], "manifest": str((output / "manifest.json").resolve()), "data_yaml": str((output / "data.yaml").resolve()), "training_command_draft": training_command, "planned_run_directory": str(run_dir.resolve()), "source_dataset_b_labels_used": False, "dataset_a_used": False, "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    (output / "staging_validation_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not result["valid"]:
        raise RuntimeError("Final-path staged data validation failed: " + "; ".join(result["errors"][:20]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
