"""Read-only validation and visual gallery for the manually labeled UAVSwarm pilot."""
from __future__ import annotations

import argparse
import html
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
DATASET_B = PROJECT / "datasets/UAVSwarm-dataset-master/UAVSwarm-dataset-master"
DATASET_A = PROJECT / "datasets/detection/drone_swarm_yolo"


def read_annotation(path: Path, image_id: str) -> dict:
    """Match annotation tool semantics: absent annotation means unreviewed, not empty-reviewed."""
    if not path.exists():
        return {"image_id": image_id, "status": "unreviewed", "boxes": []}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    fields = line.split()
    if len(fields) != 5 or fields[0] != "0":
        raise ValueError("expected class 0 and exactly four coordinates")
    try:
        vals = [float(v) for v in fields[1:]]
    except ValueError as exc:
        raise ValueError("coordinates must be numeric") from exc
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("coordinates must be finite")
    cx, cy, w, h = vals
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError("coordinates must be normalized and dimensions positive")
    if cx - w / 2 < -1e-8 or cy - h / 2 < -1e-8 or cx + w / 2 > 1 + 1e-8 or cy + h / 2 > 1 + 1e-8:
        raise ValueError("box extends outside image boundaries")
    return (0, *vals)


def discover_pilot() -> Path:
    candidates = []
    for p in (PROJECT / "datasets").rglob("manifest.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("schema_version") == 1 and data.get("class_names") == ["drone"] and isinstance(data.get("images"), list):
                if data.get("selection") and data.get("selection", {}).get("frames_per_sequence") and data.get("images") and "UAVSwarm" in data["images"][0].get("source_path", ""):
                    candidates.append(p.parent)
        except Exception:
            continue
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one matching pilot manifest, found {len(candidates)}: {candidates}")
    return candidates[0]


def audit_pilot(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("images", [])
    errors: list[str] = []
    warnings: list[str] = []
    ids = [x.get("image_id") for x in items]
    for dup in sorted(k for k, n in Counter(ids).items() if n > 1):
        errors.append(f"duplicate manifest image_id: {dup}")
    by_split = Counter(x.get("split") for x in items)
    train_seq = {x.get("sequence_id") for x in items if x.get("split") == "train"}
    val_seq = {x.get("sequence_id") for x in items if x.get("split") == "val"}
    overlap = sorted(train_seq & val_seq)
    if overlap:
        errors.append(f"train/validation sequence overlap: {overlap}")
    selection = manifest.get("selection", {})
    if selection.get("official_test_sequences_used") is not False:
        errors.append("manifest does not affirm official test sequences were excluded")
    declared_train = set(selection.get("train_sequences", []))
    declared_val = set(selection.get("validation_sequences", []))
    if declared_train != train_seq or declared_val != val_seq:
        errors.append("manifest selected sequence lists do not match image entries")
    if len(items) != 120 or by_split != Counter({"train": 90, "val": 30}):
        errors.append(f"manifest counts unexpected: total={len(items)}, splits={dict(by_split)}")

    expected_ann: set[Path] = set()
    expected_labels: set[Path] = set()
    statuses = Counter()
    split_boxes = Counter()
    records = []
    for item in items:
        image_id, split = item.get("image_id", ""), item.get("split", "")
        if split not in ("train", "val"):
            errors.append(f"{image_id}: invalid split {split!r}")
            continue
        rel = Path(item.get("source_path", ""))
        source = (PROJECT / rel).resolve()
        train_root = (DATASET_B / "train").resolve()
        test_root = (DATASET_B / "test").resolve()
        try:
            source.relative_to(train_root)
        except ValueError:
            errors.append(f"{image_id}: source is not under official train directory: {rel}")
        try:
            source.relative_to(test_root)
            errors.append(f"{image_id}: official test image used: {rel}")
        except ValueError:
            pass
        seq_from_path = source.parent.parent.name
        if item.get("sequence_id") != seq_from_path:
            errors.append(f"{image_id}: sequence_id {item.get('sequence_id')} disagrees with source path {seq_from_path}")
        if source.stem.isdigit() and int(source.stem) != item.get("frame_number"):
            errors.append(f"{image_id}: frame number disagrees with source filename")
        if not source.is_file():
            errors.append(f"{image_id}: source image missing: {source}")
        else:
            try:
                with Image.open(source) as im:
                    im.verify()
                with Image.open(source) as im:
                    dimensions = im.size
                if dimensions != (item.get("width"), item.get("height")):
                    errors.append(f"{image_id}: image dimensions {dimensions} differ from manifest {(item.get('width'), item.get('height'))}")
            except Exception as exc:
                errors.append(f"{image_id}: unreadable image: {exc}")
        ann_path = root / "annotations" / split / f"{image_id}.json"
        label_path = root / "labels" / split / f"{image_id}.txt"
        expected_ann.add(ann_path.resolve())
        expected_labels.add(label_path.resolve())
        try:
            annotation = read_annotation(ann_path, image_id)
        except Exception as exc:
            errors.append(f"{image_id}: annotation JSON invalid: {exc}")
            continue
        status = annotation.get("status")
        statuses[(split, status)] += 1
        if annotation.get("image_id") != image_id:
            errors.append(f"{image_id}: annotation image_id mismatch")
        if status not in ("reviewed", "draft", "unreviewed"):
            errors.append(f"{image_id}: invalid status {status!r}")
        boxes = annotation.get("boxes")
        if not isinstance(boxes, list):
            errors.append(f"{image_id}: boxes must be a list")
            boxes = []
        if status != "reviewed":
            errors.append(f"{image_id}: annotation is {status}, expected completed reviewed status")
        if not label_path.is_file():
            errors.append(f"{image_id}: YOLO label file missing")
            lines = []
        else:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(boxes):
            errors.append(f"{image_id}: {len(boxes)} JSON boxes vs {len(lines)} YOLO rows")
        parsed = []
        for n, line in enumerate(lines, 1):
            try:
                parsed.append(parse_yolo_line(line))
            except ValueError as exc:
                errors.append(f"{image_id} YOLO row {n}: {exc}")
        if len(set(line.strip() for line in lines)) != len(lines):
            errors.append(f"{image_id}: duplicate identical YOLO rows")
        json_seen = set()
        w_img, h_img = item.get("width", 0), item.get("height", 0)
        for n, box in enumerate(boxes):
            if not isinstance(box, dict):
                errors.append(f"{image_id} JSON box {n+1}: expected object")
                continue
            if box.get("class_id", 0) != 0:
                errors.append(f"{image_id} JSON box {n+1}: class ID is not 0")
            try:
                xyxy = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
                if not all(math.isfinite(v) for v in xyxy):
                    raise ValueError("non-finite coordinate")
                x1, y1, x2, y2 = xyxy
                if not (0 <= x1 < x2 <= w_img and 0 <= y1 < y2 <= h_img):
                    errors.append(f"{image_id} JSON box {n+1}: invalid dimensions or out of image bounds {xyxy}")
                key = tuple(round(v, 5) for v in xyxy)
                if key in json_seen:
                    errors.append(f"{image_id}: duplicate JSON box coordinates {xyxy}")
                json_seen.add(key)
                if n < len(parsed) and w_img and h_img:
                    _, cx, cy, bw, bh = parsed[n]
                    decoded = ((cx-bw/2)*w_img, (cy-bh/2)*h_img, (cx+bw/2)*w_img, (cy+bh/2)*h_img)
                    if max(abs(a-b) for a, b in zip(xyxy, decoded)) > 0.51:
                        errors.append(f"{image_id} box {n+1}: JSON/YOLO coordinates disagree by >0.51 px")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{image_id} JSON box {n+1}: invalid coordinate data: {exc}")
        split_boxes[split] += len(boxes)
        records.append({**item, "status": status, "box_count": len(boxes), "source_abs": str(source), "label_abs": str(label_path)})

    for sub, expected, suffix, kind in (("annotations", expected_ann, ".json", "annotation JSON"), ("labels", expected_labels, ".txt", "YOLO label")):
        base = root / sub
        actual = {p.resolve() for p in base.glob("*/*" + suffix)} if base.exists() else set()
        for p in sorted(actual - expected):
            errors.append(f"orphan {kind}: {p}")
        for p in sorted(expected - actual):
            errors.append(f"missing {kind}: {p}")
    reviewed = sum(n for (s, st), n in statuses.items() if st == "reviewed")
    if reviewed != len(items):
        errors.append(f"reviewed status count {reviewed} does not match manifest count {len(items)}")
    empty_reviewed = sum(1 for r in records if r["status"] == "reviewed" and r["box_count"] == 0)
    # Audit previous progress file but never rely on it as source of truth.
    progress_path = root / "progress.json"
    progress_match = None
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress_match = progress.get("total") == len(items) and progress.get("reviewed") == reviewed
            if not progress_match:
                errors.append("progress.json counts disagree with annotation files")
        except Exception as exc:
            errors.append(f"progress.json unreadable: {exc}")
    return {"pilot_root": str(root.resolve()), "manifest": str(manifest_path.resolve()), "counts": {"manifest_images": len(items), "train_images": by_split["train"], "validation_images": by_split["val"], "reviewed": reviewed, "reviewed_empty": empty_reviewed, "boxes_by_split": dict(split_boxes), "total_boxes": sum(split_boxes.values()), "annotation_json_expected": len(expected_ann), "yolo_label_expected": len(expected_labels)}, "statuses": {f"{s}_{st}": n for (s, st), n in sorted(statuses.items())}, "sequence_isolation": {"train_sequences": sorted(train_seq), "validation_sequences": sorted(val_seq), "overlap": overlap, "test_sequences_excluded_by_source_path": all("/train/" in r.get("source_path", "").replace("\\", "/") for r in items), "manifest_test_exclusion_flag": selection.get("official_test_sequences_used") is False}, "progress_file_matches": progress_match, "records": records, "errors": errors, "warnings": warnings, "structural_valid": not errors, "visual_correctness_verified": False}


def dataset_a_summary() -> dict:
    result = {"root": str(DATASET_A.resolve()), "splits": {}, "formal_review_status": "No per-image manual review status is stored in Dataset A label files.", "previous_audit": "Prior audit is structural/sample review only; it does not establish exhaustive visual correctness."}
    for split in ("train", "val"):
        imgs = [p for p in (DATASET_A / "images" / split).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        labels = list((DATASET_A / "labels" / split).glob("*.txt"))
        boxes = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in labels)
        empty = sum(not p.read_text(encoding="utf-8").strip() for p in labels)
        result["splits"][split] = {"images": len(imgs), "label_files": len(labels), "boxes": boxes, "empty_labels": empty, "missing_labels": len({p.stem for p in imgs} - {p.stem for p in labels}), "orphan_labels": len({p.stem for p in labels} - {p.stem for p in imgs})}
    return result


def choose_gallery(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen: dict[str, dict] = {}
    for split, n in (("train", 8), ("val", 8)):
        rows = [r for r in records if r["split"] == split]
        for r in rng.sample(rows, min(n, len(rows))):
            chosen.setdefault(r["image_id"], {**r, "gallery_categories": []})["gallery_categories"].append(f"random {split}")
    ranked = sorted(records, key=lambda r: (r["box_count"], r["image_id"]))
    for r in ranked[-8:]:
        chosen.setdefault(r["image_id"], {**r, "gallery_categories": []})["gallery_categories"].append("most boxes")
    for r in ranked[:8]:
        chosen.setdefault(r["image_id"], {**r, "gallery_categories": []})["gallery_categories"].append("fewest boxes")
    for r in records:
        if r["image_id"] == "UAVSwarm-47_000082":
            chosen.setdefault(r["image_id"], {**r, "gallery_categories": []})["gallery_categories"].append("prior visual concern follow-up")
    return list(chosen.values())


def render_gallery(root: Path, records: list[dict], out: Path, seed: int) -> tuple[str, int]:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    selected = choose_gallery(records, seed)
    cards = []
    for row in selected:
        src = Path(row["source_abs"])
        with Image.open(src) as im:
            image = im.convert("RGB")
        draw = ImageDraw.Draw(image)
        label_path = Path(row["label_abs"])
        for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
            try:
                _, cx, cy, bw, bh = parse_yolo_line(line)
            except ValueError:
                continue
            w, h = image.size
            x1, y1, x2, y2 = (int(round((cx-bw/2)*w)), int(round((cy-bh/2)*h)), int(round((cx+bw/2)*w)), int(round((cy+bh/2)*h)))
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 185), width=max(2, round(min(w,h)/300)))
            text = "drone 0"
            try:
                font = ImageFont.truetype("arial.ttf", max(12, round(min(w,h)/30)))
            except OSError:
                font = ImageFont.load_default()
            tb = draw.textbbox((x1, max(0, y1-20)), text, font=font)
            draw.rectangle(tb, fill=(0, 0, 0))
            draw.text((x1, max(0, y1-20)), text, fill=(0, 255, 185), font=font)
        filename = f"{row['split']}_{row['image_id']}.jpg"
        image.save(gallery / filename, quality=95, subsampling=0)
        cats = ", ".join(row["gallery_categories"])
        cards.append(f'<article><h2>{html.escape(row["image_id"])} — {html.escape(row["split"])}</h2><p>{html.escape(cats)} · {row["box_count"]} boxes · {html.escape(row["status"])}</p><p><code>{html.escape(row["source_path"])}</code></p><img src="gallery/{html.escape(filename)}" alt="Original image with manually annotated drone boxes"></article>')
    index = out / "visual_review.html"
    index.write_text("<!doctype html><meta charset='utf-8'><title>Phase 3B annotation review</title><style>body{font:15px system-ui;background:#101820;color:#e8f0f6;margin:24px}article{background:#1d2a35;padding:16px;margin:20px 0}img{display:block;max-width:100%;height:auto;border:1px solid #526477}code{overflow-wrap:anywhere}</style><h1>Phase 3B visual sample</h1><p>All rendered source images are original resolution, with final YOLO labels overlaid. Sample is reproducible; manual visual assessment is required.</p>" + "\n".join(cards), encoding="utf-8")
    return str(index.resolve()), len(selected)


def write_report(out: Path, audit: dict, dataset_a: dict, seed: int) -> None:
    safe = {k: v for k, v in audit.items() if k != "records"}
    safe["gallery_seed"] = seed
    safe["sampled_visual_assessment"] = {
        "observed": "This report contains a reproducible sample gallery with final YOLO boxes overlaid on original-resolution images. Structural validation does not establish visual correctness for every image.",
        "interpretation": "Review sampled overlays and the append-only adjudication log. Human annotation correctness is not asserted for uninspected images.",
        "human_visual_correctness_verified_for_entire_dataset": False,
    }
    safe["dataset_a"] = dataset_a
    safe["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "audit_report.json").write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    c = audit["counts"]
    status = "PASS" if audit["structural_valid"] else "FAIL"
    lines = ["# Phase 3B annotation validation", "", f"Structural result: **{status}**", "", f"Pilot: `{audit['pilot_root']}`", f"Manifest: `{audit['manifest']}`", "", f"Manifest images: {c['manifest_images']} (train {c['train_images']}, validation {c['validation_images']})", f"Reviewed: {c['reviewed']}; reviewed-empty: {c['reviewed_empty']}; boxes: {c['total_boxes']} (train {c['boxes_by_split'].get('train',0)}, validation {c['boxes_by_split'].get('val',0)})", f"YOLO files / JSON files expected: {c['yolo_label_expected']} / {c['annotation_json_expected']}", f"Train/val sequence overlap: {audit['sequence_isolation']['overlap']}", f"Official test excluded by manifest/path checks: {audit['sequence_isolation']['test_sequences_excluded_by_source_path'] and audit['sequence_isolation']['manifest_test_exclusion_flag']}", f"Progress file agrees: {audit['progress_file_matches']}", "", "## Errors", ""]
    lines += [f"- {x}" for x in audit["errors"]] or ["- None"]
    lines += ["", "## Visual review", "", "Open `visual_review.html`. This is a seeded sample gallery with final YOLO boxes overlaid on original-resolution images. Structural validation does not establish visual correctness for every image; inspect the gallery and adjudication log before using labels beyond the reviewed cases.", "", "## Dataset A status", "", f"Dataset A root: `{dataset_a['root']}`", ""]
    for split, data in dataset_a["splits"].items():
        lines.append(f"- {split}: {data['images']} images, {data['label_files']} label files, {data['boxes']} boxes; missing labels {data['missing_labels']}, orphan labels {data['orphan_labels']}, empty labels {data['empty_labels']}.")
    lines += ["", dataset_a["formal_review_status"], dataset_a["previous_audit"], "", "No source image, annotation, label, checkpoint, or prior output was modified.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(out: Path, seed: int = 20261027) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    root = discover_pilot()
    audit = audit_pilot(root)
    gallery, n = render_gallery(root, audit["records"], out, seed)
    audit["gallery_path"] = gallery
    audit["gallery_images"] = n
    a = dataset_a_summary()
    write_report(out, audit, a, seed)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, ensure_ascii=False))
    if not result["structural_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
