#!/usr/bin/env python3
"""Read-only structural and visual audit for the two drone datasets.

Requires Pillow for image dimensions and drawing. Source datasets are opened
read-only; all generated reports/images are written to a fresh output folder.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("Pillow is required. Use the bundled Python runtime or install Pillow separately after approval.") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCRIPT_VERSION = 2


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.as_posix().lower())


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in dict.fromkeys(indices)]


def parse_yolo(label: Path) -> tuple[list[dict[str, Any]], list[str]]:
    boxes: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = label.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        return boxes, [f"label_read_error:{exc}"]
    for line_no, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"line_{line_no}:expected_5_columns_got_{len(fields)}")
            continue
        try:
            class_float = float(fields[0])
            values = [float(x) for x in fields[1:]]
            if not math.isfinite(class_float) or not class_float.is_integer():
                raise ValueError("class_id_not_integer")
            if not all(math.isfinite(x) for x in values):
                raise ValueError("non_finite_coordinate")
            cx, cy, width, height = values
            boxes.append({"class_id": int(class_float), "cx": cx, "cy": cy, "w": width, "h": height,
                          "x1": cx - width / 2, "y1": cy - height / 2,
                          "x2": cx + width / 2, "y2": cy + height / 2})
            if width <= 0 or height <= 0:
                errors.append(f"line_{line_no}:nonpositive_box_size")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                errors.append(f"line_{line_no}:normalized_value_out_of_range")
            if cx - width / 2 < 0 or cy - height / 2 < 0 or cx + width / 2 > 1 or cy + height / 2 > 1:
                errors.append(f"line_{line_no}:box_exceeds_normalized_image_bounds")
        except (ValueError, OverflowError) as exc:
            errors.append(f"line_{line_no}:{exc}")
    return boxes, errors


def draw_yolo_overlay(image_path: Path, boxes: list[dict[str, Any]], output: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    iw, ih = image.size
    for box in boxes:
        xy = (round(box["x1"] * iw), round(box["y1"] * ih), round(box["x2"] * iw), round(box["y2"] * ih))
        draw.rectangle(xy, outline=(0, 255, 80), width=max(2, min(iw, ih) // 400))
        draw.text((xy[0], max(0, xy[1] - 12)), f"class {box['class_id']}", fill=(0, 255, 80), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    image.save(output, quality=92)


def audit_dataset_a(root: Path, out: Path, sample_count: int, seed: int) -> dict[str, Any]:
    base = root / "datasets" / "detection" / "drone_swarm_yolo"
    image_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    selected: list[tuple[str, Path, Path, list[dict[str, Any]]]] = []
    rng = random.Random(seed)

    for split in ("train", "val"):
        images = image_files(base / "images" / split)
        labels_dir = base / "labels" / split
        by_stem: dict[str, list[Path]] = defaultdict(list)
        for label in labels_dir.glob("*.txt") if labels_dir.is_dir() else []:
            by_stem[label.stem].append(label)
        image_stems = Counter(p.stem for p in images)
        missing_labels = []
        all_boxes: list[dict[str, Any]] = []
        split_sizes: Counter[str] = Counter()
        for image_path in images:
            matches = by_stem.get(image_path.stem, [])
            size = image_size(image_path)
            if size is None:
                image_rows.append({"split": split, "image": image_path.relative_to(root).as_posix(), "width": "", "height": "", "boxes": "", "status": "unreadable_image"})
                continue
            width_px, height_px = size
            split_sizes[f"{width_px}x{height_px}"] += 1
            if len(matches) != 1:
                missing_labels.append(image_path.relative_to(root).as_posix())
                image_rows.append({"split": split, "image": image_path.relative_to(root).as_posix(), "width": width_px, "height": height_px, "boxes": 0, "status": "missing_label" if not matches else "duplicate_label_stem"})
                continue
            boxes, errors = parse_yolo(matches[0])
            all_boxes.extend(boxes)
            image_rows.append({"split": split, "image": image_path.relative_to(root).as_posix(), "width": width_px, "height": height_px,
                               "boxes": len(boxes), "status": ";".join(errors) or "ok"})

        label_stems = set(by_stem)
        image_stem_set = set(image_stems)
        orphan_labels = sorted(label_stems - image_stem_set)
        candidates = [p for p in images if len(by_stem.get(p.stem, [])) == 1]
        picked = sorted(rng.sample(candidates, min(sample_count, len(candidates))), key=lambda p: p.as_posix().lower())
        for image_path in picked:
            boxes, _ = parse_yolo(by_stem[image_path.stem][0])
            selected.append((split, image_path, by_stem[image_path.stem][0], boxes))

        widths = [b["w"] for b in all_boxes if b["w"] > 0]
        heights = [b["h"] for b in all_boxes if b["h"] > 0]
        split_summary[split] = {
            "images": len(images), "label_files": sum(len(v) for v in by_stem.values()),
            "missing_or_ambiguous_labels": missing_labels, "orphan_label_files": orphan_labels,
            "dimension_counts": dict(split_sizes), "box_count": len(all_boxes),
            "class_counts": dict(Counter(str(b["class_id"]) for b in all_boxes)),
            "normalized_box_width": distribution(widths), "normalized_box_height": distribution(heights),
        }

    images_out = out / "dataset_a"
    images_out.mkdir(parents=True, exist_ok=True)
    gallery = []
    for split, image_path, label_path, boxes in selected:
        name = f"{split}_{image_path.stem}_overlay.png"
        draw_yolo_overlay(image_path, boxes, images_out / name)
        gallery.append({"split": split, "image": image_path.relative_to(root).as_posix(),
                        "label": label_path.relative_to(root).as_posix(), "overlay": f"dataset_a/{name}",
                        "width": image_size(image_path)[0], "height": image_size(image_path)[1], "boxes": len(boxes)})

    with (out / "dataset_a_images.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "width", "height", "boxes", "status"])
        writer.writeheader()
        writer.writerows(image_rows)
    write_gallery(out / "dataset_a_review.html", "Dataset A — YOLO ground-truth review", gallery,
                  "Boxes are drawn from the existing YOLO label files. This is a visual aid; automated checks do not establish that a box encloses a drone.")
    return {"path": base.relative_to(root).as_posix(), "split_summary": split_summary,
            "visual_samples": len(gallery), "automated_checks_only": True}


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def parse_ini(path: Path) -> dict[str, str]:
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", ";", "[")) and "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    return result


def parse_mot(path: Path) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    errors = []
    if not path.exists():
        return result, ["missing_annotation_file"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        fields = [v.strip() for v in re.split(r"[\s,]+", line.strip())]
        if len(fields) < 6:
            errors.append(f"line_{line_no}:too_few_columns_{len(fields)}")
            continue
        try:
            frame = int(float(fields[0]))
            track = int(float(fields[1]))
            x, y, w, h = map(float, fields[2:6])
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError("non_finite_bbox")
            result[frame].append({"track_id": track, "x": x, "y": y, "w": w, "h": h, "line": line_no})
            if len(fields) != 9:
                errors.append(f"line_{line_no}:expected_9_columns_got_{len(fields)}")
        except (ValueError, OverflowError) as exc:
            errors.append(f"line_{line_no}:{exc}")
    return result, errors


def bounds_state(box: dict[str, Any], width: int, height: int) -> str:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    if w <= 0 or h <= 0:
        return "invalid_size"
    x2, y2 = x + w, y + h
    if x >= 0 and y >= 0 and x2 <= width and y2 <= height:
        return "inside"
    if x2 <= 0 or y2 <= 0 or x >= width or y >= height:
        return "outside"
    return "partial"


def draw_mot_overlay(image_path: Path, boxes: list[dict[str, Any]], output: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    iw, ih = image.size
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        state = bounds_state(box, iw, ih)
        color = {"inside": (0, 255, 80), "partial": (255, 190, 0), "outside": (255, 50, 50), "invalid_size": (255, 50, 255)}[state]
        x1, y1, x2, y2 = round(x), round(y), round(x + w), round(y + h)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, min(iw, ih) // 400))
        draw.text((x1, max(0, y1 - 12)), f"ID {box['track_id']} {state}", fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    image.save(output, quality=92)


def json_split_summary(base: Path, split: str) -> dict[str, Any]:
    name = "train.json" if split == "train" else "test.json"
    path = base / "annotations" / name
    if not path.exists():
        return {"file": name, "present": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"file": name, "present": True, "error": str(exc)}
    images = data.get("images", [])
    anns = data.get("annotations", [])
    id_to_image = {x.get("id"): x for x in images}
    video_map = {x.get("id"): x.get("file_name") for x in data.get("videos", [])}
    missing_references = 0
    for item in images:
        parts = Path(str(item.get("file_name", ""))).parts
        seq = parts[0] if parts else ""
        frame = Path(parts[-1]).stem if len(parts) >= 2 else ""
        image_path = base / split / seq / "img1" / f"{frame}.jpg"
        if not image_path.exists():
            # Keep original extension if it was not .jpg.
            image_path = base / split / Path(*parts)
        if not image_path.exists():
            missing_references += 1
    category_names = [x.get("name") for x in data.get("categories", [])]
    return {"file": name, "present": True, "images": len(images), "annotations": len(anns),
            "videos": len(data.get("videos", [])), "categories": category_names,
            "missing_image_references": missing_references,
            "sample_video_names": list(video_map.values())[:3],
            "annotation_image_id_missing": sum(1 for a in anns if a.get("image_id") not in id_to_image)}


def compare_coco_mot(base: Path, split: str, mot_rows: Counter[tuple[Any, ...]],
                     mot_track_by_box: dict[tuple[Any, ...], Counter[int]]) -> dict[str, Any]:
    """Compare MOT GT rows with the bundled COCO-style annotation JSON."""
    name = "train.json" if split == "train" else "test.json"
    path = base / "annotations" / name
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {item.get("id"): item for item in data.get("images", [])}
    coco_rows: Counter[tuple[Any, ...]] = Counter()
    coco_track_by_box: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    unresolvable = 0
    for ann in data.get("annotations", []):
        item = images.get(ann.get("image_id"))
        bbox = ann.get("bbox")
        if item is None or not isinstance(bbox, list) or len(bbox) != 4:
            unresolvable += 1
            continue
        parts = Path(str(item.get("file_name", ""))).parts
        try:
            sequence = parts[0]
            frame = int(Path(parts[-1]).stem)
            track = int(ann.get("track_id", -1))
            x, y, w, h = (round(float(value), 4) for value in bbox)
            key = (sequence, frame, x, y, w, h)
            coco_rows[key] += 1
            coco_track_by_box[key][track] += 1
        except (ValueError, TypeError, IndexError):
            unresolvable += 1
    matching = sum((mot_rows & coco_rows).values())
    mot_count, coco_count = sum(mot_rows.values()), sum(coco_rows.values())
    track_pairs: Counter[tuple[str, int, int]] = Counter()
    for key, amount in (mot_rows & coco_rows).items():
        seq = key[0]
        json_ids = coco_track_by_box[key]
        mot_ids = mot_track_by_box.get(key, Counter())
        remaining = amount
        for json_id, json_n in sorted(json_ids.items()):
            for mot_id, mot_n in sorted(mot_ids.items()):
                matched_n = min(remaining, json_n, mot_n)
                if matched_n:
                    track_pairs[(seq, json_id, mot_id)] += matched_n
                    remaining -= matched_n
                    json_n -= matched_n
                    mot_n -= matched_n
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
    json_to_mot: dict[tuple[str, int], set[int]] = defaultdict(set)
    mot_to_json: dict[tuple[str, int], set[int]] = defaultdict(set)
    for (seq, json_id, mot_id), _ in track_pairs.items():
        json_to_mot[(seq, json_id)].add(mot_id)
        mot_to_json[(seq, mot_id)].add(json_id)
    inconsistent_json_ids = sum(1 for ids in json_to_mot.values() if len(ids) != 1)
    inconsistent_mot_ids = sum(1 for ids in mot_to_json.values() if len(ids) != 1)
    return {"available": True, "mot_rows": mot_count, "json_rows": coco_count,
            "exact_matching_boxes_ignoring_track_id": matching, "mot_rows_without_exact_json_match": mot_count - matching,
            "json_rows_without_exact_mot_match": coco_count - matching,
            "unresolvable_json_rows": unresolvable,
            "exact_match_pct_of_mot": 100 * matching / max(1, mot_count),
            "matched_sequence_local_to_global_track_id_pairs": len(track_pairs),
            "json_track_ids_mapping_to_multiple_mot_ids_within_sequence": inconsistent_json_ids,
            "mot_track_ids_mapping_to_multiple_json_ids_within_sequence": inconsistent_mot_ids}


def audit_dataset_b(root: Path, out: Path, seed: int) -> dict[str, Any]:
    base = root / "datasets" / "UAVSwarm-dataset-master" / "UAVSwarm-dataset-master"
    (out / "dataset_b").mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, Any] = {}
    sequence_rows = []
    all_vis_samples: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for split in ("train", "test"):
        split_dir = base / split
        sequences = sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()) if split_dir.is_dir() else []
        total_frames = total_annotation_rows = total_frame_refs_missing = 0
        all_states: Counter[str] = Counter()
        all_errors = Counter()
        per_sequence: list[dict[str, Any]] = []
        mot_rows: Counter[tuple[Any, ...]] = Counter()
        mot_track_by_box: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)

        for seq_dir in sequences:
            meta = parse_ini(seq_dir / "seqinfo.ini")
            frames = image_files(seq_dir / "img1")
            frame_by_number: dict[int, Path] = {}
            for fp in frames:
                match = re.search(r"(\d+)$", fp.stem)
                if match:
                    frame_by_number[int(match.group(1))] = fp
            gt, errors = parse_mot(seq_dir / "gt" / "gt.txt")
            expected_len = int(meta.get("seqLength", "0") or 0)
            present_numbers = set(frame_by_number)
            expected_numbers = set(range(1, expected_len + 1)) if expected_len else set()
            missing_frames = sorted(expected_numbers - present_numbers) if expected_numbers else []
            image_dims: dict[int, tuple[int, int] | None] = {n: image_size(p) for n, p in frame_by_number.items()}
            try:
                metadata_size = (int(meta["imWidth"]), int(meta["imHeight"]))
            except (KeyError, ValueError):
                metadata_size = None
            metadata_size_mismatches = sum(1 for dims in image_dims.values() if dims is not None and metadata_size is not None and dims != metadata_size)
            counts: Counter[str] = Counter()
            box_records: list[tuple[int, dict[str, Any], str]] = []
            for frame_no, boxes in gt.items():
                dims = image_dims.get(frame_no)
                if dims is None:
                    total_frame_refs_missing += len(boxes)
                    continue
                iw, ih = dims
                for box in boxes:
                    state = bounds_state(box, iw, ih)
                    counts[state] += 1
                    all_states[state] += 1
                    box_records.append((frame_no, box, state))
                    box_key = (seq_dir.name, frame_no, round(box["x"], 4), round(box["y"], 4), round(box["w"], 4), round(box["h"], 4))
                    mot_rows[box_key] += 1
                    mot_track_by_box[box_key][box["track_id"]] += 1
            for error in errors:
                all_errors[error.split(":", 1)[0]] += 1
            row = {"split": split, "sequence": seq_dir.name, "declared_frames": expected_len or None,
                   "image_files": len(frames), "numbered_frames": len(frame_by_number),
                   "missing_expected_frames": missing_frames,
                   "numbered_frames_beyond_declared_length": sorted(n for n in present_numbers if expected_len and n > expected_len),
                   "metadata_dimensions": metadata_size,
                   "images_with_dimensions_mismatching_metadata": metadata_size_mismatches,
                   "annotation_frames": len(gt), "annotation_boxes": sum(map(len, gt.values())),
                   "annotation_rows_with_missing_frame_image": sum(len(v) for k, v in gt.items() if k not in frame_by_number),
                   "annotation_frames_outside_declared_length": sorted(k for k in gt if expected_len and (k < 1 or k > expected_len)),
                   "inside": counts["inside"], "partial": counts["partial"], "outside": counts["outside"],
                   "invalid_size": counts["invalid_size"], "parse_errors": errors,
                   "out_of_bounds_examples": [{"frame": frame_no, "track_id": box["track_id"], "x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"], "state": state}
                                               for frame_no, box, state in box_records if state != "inside"][:10],
                   "metadata": meta}
            per_sequence.append(row)
            sequence_rows.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
            total_frames += len(frames)
            total_annotation_rows += sum(map(len, gt.values()))

            # Deterministic sample sequence selection: cover first/middle/last,
            # plus the sequence with the highest measured outside/partial share.
            row["_records"] = box_records

        frame_refs = sum(r["annotation_rows_with_missing_frame_image"] for r in per_sequence)
        sequence_choices = evenly_spaced(per_sequence, min(3, len(per_sequence)))
        nonempty = [r for r in per_sequence if r["_records"]]
        if nonempty:
            worst = max(nonempty, key=lambda r: (r["outside"] + r["partial"]) / max(1, r["inside"] + r["partial"] + r["outside"] + r["invalid_size"]))
            if worst not in sequence_choices:
                sequence_choices.append(worst)

        for row in sequence_choices:
            records = row["_records"]
            by_frame: dict[int, list[tuple[dict[str, Any], str]]] = defaultdict(list)
            for frame_no, box, state in records:
                by_frame[frame_no].append((box, state))
            frame_candidates = sorted(by_frame)
            if not frame_candidates:
                continue
            median_frame = frame_candidates[len(frame_candidates) // 2]
            outside_frame = max(frame_candidates, key=lambda n: (sum(s != "inside" for _, s in by_frame[n]), len(by_frame[n]), -n))
            for frame_no, label in dict.fromkeys([(median_frame, "middle annotated frame"), (outside_frame, "frame with most out-of-bounds boxes")]):
                seq_path = split_dir / row["sequence"]
                image_path = next((p for p in image_files(seq_path / "img1") if p.stem.isdigit() and int(p.stem) == frame_no), None)
                if image_path is None:
                    continue
                overlay_name = f"{split}_{row['sequence']}_frame{frame_no:06d}_gt_overlay.png"
                boxes = [box for box, _ in by_frame[frame_no]]
                draw_mot_overlay(image_path, boxes, out / "dataset_b" / overlay_name)
                all_vis_samples.append({"split": split, "sequence": row["sequence"], "frame": frame_no, "selection": label,
                                        "image": image_path.relative_to(root).as_posix(), "overlay": f"dataset_b/{overlay_name}",
                                        "width": image_size(image_path)[0], "height": image_size(image_path)[1],
                                        "boxes": len(boxes), "outside_or_partial": sum(bounds_state(b, *image_size(image_path)) != "inside" for b in boxes)})

        total_boxes = sum(all_states.values())
        denominator = max(1, total_boxes)
        split_summary[split] = {
            "sequences": len(sequences), "image_files": total_frames, "annotation_rows": total_annotation_rows,
            "annotation_rows_without_matching_image": frame_refs,
            "box_bounds": {"fully_inside": all_states["inside"], "partially_outside": all_states["partial"],
                           "fully_outside": all_states["outside"], "invalid_nonpositive_size": all_states["invalid_size"],
                           "fully_inside_pct": 100 * all_states["inside"] / denominator,
                           "partially_outside_pct": 100 * all_states["partial"] / denominator,
                           "fully_outside_pct": 100 * all_states["outside"] / denominator},
            "parse_error_counts": dict(all_errors),
            "sequence_summaries": [{k: v for k, v in r.items() if k != "_records"} for r in per_sequence],
            "coco_json": json_split_summary(base, split),
            "coco_vs_mot_exact_box_comparison": compare_coco_mot(base, split, mot_rows, mot_track_by_box),
        }

    with (out / "dataset_b_sequences.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in sequence_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sequence_rows)
    write_gallery(out / "dataset_b_review.html", "Dataset B — supplied MOT annotation review", all_vis_samples,
                  "Green boxes are fully inside the image, amber boxes cross an image boundary, and red boxes lie fully outside. These are supplied annotations, not verified ground truth. Check visible objects and possible frame alignment manually.")
    return {"path": base.relative_to(root).as_posix(), "readme": (base / "README.md").relative_to(root).as_posix(),
            "split_summary": split_summary, "visual_samples": len(all_vis_samples), "annotations_trusted": False}


def write_gallery(path: Path, title: str, items: list[dict[str, Any]], note: str) -> None:
    cards = []
    for item in items:
        details = " · ".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in item.items() if k not in {"overlay"})
        cards.append(f'<figure><img src="{html.escape(item["overlay"])}" alt="annotated sample"><figcaption>{details}</figcaption></figure>')
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font:15px/1.45 system-ui,sans-serif;margin:2rem;background:#f4f6f8;color:#17202a}}h1{{margin-bottom:.4rem}}.note{{max-width:1000px;padding:1rem;background:#fff3cd;border-left:4px solid #d99b00}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem;margin-top:1.5rem}}figure{{margin:0;padding:.7rem;background:white;box-shadow:0 1px 5px #0002}}img{{display:block;width:100%;height:auto}}figcaption{{font-size:.83rem;overflow-wrap:anywhere;margin-top:.5rem}}
</style></head><body><h1>{html.escape(title)}</h1><p class="note">{html.escape(note)}</p><main>{''.join(cards)}</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def choose_output(parent: Path, requested: Path | None) -> Path:
    if requested:
        path = requested.resolve()
        if path.exists() and any(path.iterdir()):
            raise SystemExit(f"Refusing to write into non-empty output directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return path
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = parent / f"run_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root (default: parent of tools/)")
    parser.add_argument("--output", type=Path, help="New or empty output directory. Default chooses outputs/dataset_audit_phase2a/run_NNN.")
    parser.add_argument("--sample-per-split", type=int, default=12, help="Dataset A overlay samples per split (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible Dataset A sample seed (default: 42)")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.sample_per_split < 0:
        parser.error("--sample-per-split must be nonnegative")
    out = choose_output(root / "outputs" / "dataset_audit_phase2a", args.output)
    report = {"script_version": SCRIPT_VERSION, "root": str(root), "seed": args.seed,
              "notes": ["Automated checks are structural only.", "Overlay generation does not verify that annotations match visible drones.", "Dataset B supplied annotations remain untrusted until manual alignment review."],
              "dataset_a": audit_dataset_a(root, out, args.sample_per_split, args.seed),
              "dataset_b": audit_dataset_b(root, out, args.seed)}
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.txt").write_text(
        "Generated by tools/audit_datasets.py. Source datasets were read only.\n"
        "Open dataset_a_review.html and dataset_b_review.html for original-resolution overlays.\n"
        "The JSON/CSV checks are automated structural checks, not proof of visual correctness.\n"
        "Each run uses a fresh output directory to avoid overwriting earlier audit files.\n",
        encoding="utf-8")
    print(f"Audit complete: {out}")
    print(f"Dataset A review: {out / 'dataset_a_review.html'}")
    print(f"Dataset B review: {out / 'dataset_b_review.html'}")
    print(f"Report: {out / 'audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
