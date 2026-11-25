"""Safely stage Seraphim's official train split with the reviewed real pilot.

This utility never reads or uses Seraphim's test split. It refuses to overwrite
an existing output directory, validates archive/label correspondence before
extraction, and writes a YOLO dataset manifest that references source files
without copying the existing reviewed pilot images.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVES = PROJECT / "datasets/external_seraphim_detector_v1/archives/train"
DEFAULT_OUTPUT = PROJECT / "datasets/external_seraphim_detector_v1/prepared"
DEFAULT_PILOT = PROJECT / "datasets/derived_swarm_training_v2"
IMAGE_ARCHIVES = [f"images/batch_{i:03d}.zip" for i in range(1, 5)]
LABEL_ARCHIVE = "labels/batch_001.zip"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def safe_member_name(name: str) -> str:
    """Return a flat archive member name, rejecting traversal and directories."""
    p = PurePosixPath(name)
    if "\\" in name or ":" in name or p.is_absolute() or ".." in p.parts or len(p.parts) != 1 or p.name != name:
        raise ValueError(f"Unexpected non-flat or unsafe archive path: {name!r}")
    return name


def archive_members(path: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as zf:
        result = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = safe_member_name(info.filename)
            if name in result:
                raise ValueError(f"Duplicate member {name!r} in {path}")
            result[name] = info
    return result


def parse_yolo(text: str, label_file: str) -> tuple[int, list[str], list[str], list[dict]]:
    count = 0
    errors: list[str] = []
    output_lines: list[str] = []
    conversions: list[dict] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5 and not (len(parts) >= 7 and (len(parts) - 1) % 2 == 0):
            errors.append(f"{label_file}:{line_no}: expected 5 bbox fields or one YOLO polygon, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{label_file}:{line_no}: non-numeric YOLO row")
            continue
        if cls != 0:
            errors.append(f"{label_file}:{line_no}: class {cls}, expected 0")
            continue
        if not all(math.isfinite(x) for x in vals):
            errors.append(f"{label_file}:{line_no}: non-finite coordinate")
            continue
        if len(parts) == 5:
            cx, cy, w, h = vals
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                errors.append(f"{label_file}:{line_no}: values outside normalized ranges")
                continue
            if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
                errors.append(f"{label_file}:{line_no}: box extends beyond image bounds")
                continue
            output_lines.append(" ".join(parts))
        else:
            xs, ys = vals[0::2], vals[1::2]
            if not all(0 <= x <= 1 for x in vals):
                errors.append(f"{label_file}:{line_no}: polygon coordinates outside normalized ranges")
                continue
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            if xmax <= xmin or ymax <= ymin:
                errors.append(f"{label_file}:{line_no}: polygon has zero-area extent")
                continue
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin
            converted_line = f"0 {cx:.12f} {cy:.12f} {w:.12f} {h:.12f}"
            output_lines.append(converted_line)
            conversions.append({
                "label_file": label_file,
                "row": line_no,
                "method": "polygon_min_max_to_yolo_box",
                "vertices": len(xs),
                "original_row": raw.strip(),
                "converted_row": converted_line,
                "bbox_xyxy_normalized": [xmin, ymin, xmax, ymax],
            })
        count += 1
    return count, errors, output_lines, conversions


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_zip(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = safe_member_name(info.filename)
            target = dest / name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite extracted file: {target}")
            with zf.open(info) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    args = parser.parse_args()
    archives = args.archives.resolve()
    output = args.output.resolve()
    pilot = args.pilot.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing prepared dataset: {output}")
    if not (archives / LABEL_ARCHIVE).is_file():
        raise FileNotFoundError(f"Missing training labels archive: {archives / LABEL_ARCHIVE}")
    for rel in IMAGE_ARCHIVES:
        if not (archives / rel).is_file():
            raise FileNotFoundError(f"Missing training images archive: {archives / rel}")
    if not (pilot / "data.yaml").is_file() or not (pilot / "manifest.json").is_file():
        raise FileNotFoundError(f"Expected validated pilot staging is missing: {pilot}")
    pilot_manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    pilot_rows = pilot_manifest.get("images", [])
    pilot_train = [r for r in pilot_rows if r.get("split") == "train"]
    pilot_val = [r for r in pilot_rows if r.get("split") == "val"]
    if len(pilot_train) != 90 or len(pilot_val) != 30:
        raise ValueError(f"Pilot split must be 90/30; found train={len(pilot_train)}, val={len(pilot_val)}")

    report: dict = {
        "source": "lgrzybowski/seraphim-drone-detection-dataset",
        "split_used": "train only",
        "official_test_split_used_for_training": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive_root": str(archives),
        "pilot_root": str(pilot),
        "pilot_train_images": len(pilot_train),
        "pilot_validation_images": len(pilot_val),
        "errors": [],
        "seraphim_images": 0,
        "seraphim_labels": 0,
        "seraphim_boxes": 0,
        "empty_label_files": 0,
        "converted_polygon_rows": 0,
        "images_by_archive": {},
        "box_count_distribution": {},
    }

    image_names: dict[str, str] = {}
    for rel in IMAGE_ARCHIVES:
        path = archives / rel
        members = archive_members(path)
        counts = Counter()
        for name in members:
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                report["errors"].append(f"Unexpected non-image member: {rel}/{name}")
            if name in image_names:
                report["errors"].append(f"Duplicate image basename across archives: {name}")
            image_names[name] = rel
            counts[Path(name).suffix.lower()] += 1
        report["images_by_archive"][rel] = {"count": len(members), "extensions": dict(counts), "bytes_uncompressed": sum(i.file_size for i in members.values())}
        report["seraphim_images"] += len(members)

    label_path = archives / LABEL_ARCHIVE
    label_members = archive_members(label_path)
    label_names = set(label_members)
    expected_labels = {f"{Path(n).stem}.txt" for n in image_names}
    missing = sorted(expected_labels - label_names)
    orphan = sorted(label_names - expected_labels)
    if missing:
        report["errors"].append(f"Missing label files: {len(missing)}; examples={missing[:10]}")
    if orphan:
        report["errors"].append(f"Orphan label files: {len(orphan)}; examples={orphan[:10]}")
    report["seraphim_labels"] = len(label_members)
    boxes = Counter()
    converted_labels: dict[str, str] = {}
    conversions: list[dict] = []
    with zipfile.ZipFile(label_path) as zf:
        for name, info in label_members.items():
            count, errors, output_lines, row_conversions = parse_yolo(zf.read(info).decode("utf-8-sig"), name)
            boxes[count] += 1
            report["seraphim_boxes"] += count
            if count == 0:
                report["empty_label_files"] += 1
            converted_labels[name] = "\n".join(output_lines) + ("\n" if output_lines else "")
            conversions.extend(row_conversions)
            report["errors"].extend(errors[: max(0, 30 - len(report["errors"]))])
    report["box_count_distribution"] = {str(k): v for k, v in sorted(boxes.items())}
    report["converted_polygon_rows"] = len(conversions)
    if report["errors"]:
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = output.parent / f"{output.name}_source_audit_failed.json"
        diagnostic.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise ValueError(f"Seraphim archive/label audit failed; details: {diagnostic}")

    output.mkdir(parents=True, exist_ok=False)
    images_out = output / "extracted/train/images"
    labels_out = output / "extracted/train/labels"
    for rel in IMAGE_ARCHIVES:
        extract_zip(archives / rel, images_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    for name, text in converted_labels.items():
        (labels_out / name).write_text(text, encoding="utf-8", newline="\n")

    # Verify readable images and final-path label/image correspondence.
    extracted_images = sorted(p for p in images_out.iterdir() if p.is_file())
    extracted_labels = {p.name for p in labels_out.glob("*.txt")}
    if len(extracted_images) != report["seraphim_images"]:
        raise RuntimeError(f"Extracted image count mismatch: {len(extracted_images)}")
    if extracted_labels != expected_labels:
        raise RuntimeError(f"Extracted label count/name mismatch: labels={len(extracted_labels)} expected={len(expected_labels)}")
    unreadable = []
    for image in extracted_images:
        try:
            with Image.open(image) as im:
                im.verify()
        except Exception as exc:
            unreadable.append(f"{image.name}: {exc}")
            if len(unreadable) >= 20:
                break
    if unreadable:
        report["errors"] = unreadable
        (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Unreadable images found; see {output / 'preparation_report.json'}")

    # Keep the source polygons and all conversions auditable. The original ZIP
    # stays unchanged; previews show the polygon outline and derived bbox.
    conversion_dir = output / "label_conversion_review"
    conversion_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(conversions, 1):
        filename = item["label_file"].removesuffix(".txt") + ".jpg"
        image_path = images_out / filename
        with Image.open(image_path) as source:
            preview = source.convert("RGB")
        draw = ImageDraw.Draw(preview)
        raw_values = [float(x) for x in item["original_row"].split()[1:]]
        points = [(raw_values[j] * preview.width, raw_values[j + 1] * preview.height) for j in range(0, len(raw_values), 2)]
        draw.line(points + [points[0]], fill=(255, 0, 255), width=2)
        coords = item["bbox_xyxy_normalized"]
        x1, y1, x2, y2 = (coords[0] * preview.width, coords[1] * preview.height, coords[2] * preview.width, coords[3] * preview.height)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        draw.text((x1, max(0, y1 - 14)), f"derived box row {item['row']}", fill=(0, 255, 0))
        preview.save(conversion_dir / f"polygon_{i:03d}_{filename}", quality=95)
        item["preview"] = str((conversion_dir / f"polygon_{i:03d}_{filename}").relative_to(output).as_posix())
    (output / "label_conversion_audit.json").write_text(json.dumps(conversions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # YOLO accepts text manifests of absolute image paths; this avoids copying
    # 75k images and keeps source labels beside their original source images.
    pilot_image_root = pilot / "images"
    train_paths = [str((images_out / name).resolve()).replace("\\", "/") for name in sorted(image_names)]
    train_paths.extend(str((pilot / r["staged_image"]).resolve()).replace("\\", "/") for r in pilot_train)
    val_paths = [str((pilot / r["staged_image"]).resolve()).replace("\\", "/") for r in pilot_val]
    if not all(Path(p).is_file() for p in train_paths[-len(pilot_train):] + val_paths):
        raise FileNotFoundError(f"Pilot paths missing from staged root: {pilot_image_root}")
    (output / "train_images.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8", newline="\n")
    (output / "val_images.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8", newline="\n")
    yaml_text = (
        f'path: "{output.as_posix()}"\n'
        "train: train_images.txt\n"
        "val: val_images.txt\n"
        "nc: 1\n"
        "names:\n  0: drone\n"
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8", newline="\n")
    report.update({
        "prepared_root": str(output),
        "extracted_train_images": len(extracted_images),
        "combined_train_images": len(train_paths),
        "real_validation_images": len(val_paths),
        "class_names": ["drone"],
        "training_data_sources": ["Seraphim official train split", "manually reviewed 90-image pilot train split"],
        "validation_data_source": "manually reviewed 30-image pilot validation split",
        "dataset_yaml": str(output / "data.yaml"),
        "test_split_used_for_training_or_validation": False,
        "archive_sha256": {p.relative_to(archives).as_posix(): digest(p) for p in [*(archives / r for r in IMAGE_ARCHIVES), label_path]},
        "errors": [],
    })
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
