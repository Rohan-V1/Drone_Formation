#!/usr/bin/env python3
"""Reproducible, read-only-source review of an existing YOLO checkpoint.

The script copies Dataset A validation files into its fresh output directory
before Ultralytics validation so that any generated cache stays out of source
datasets. Inference and overlay outputs also stay under that output directory.
It never trains and refuses to reuse a nonempty output directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import shutil
from pathlib import Path
from statistics import median
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_yolo(path: Path) -> list[dict[str, float | int]]:
    boxes: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        cls, cx, cy, w, h = fields
        boxes.append({"class_id": int(float(cls)), "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)})
    return boxes


def draw_gt(image_path: Path, boxes: list[dict[str, float | int]], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as im:
        image = im.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    line_width = max(2, min(width, height) // 400)
    for box in boxes:
        cx, cy, bw, bh = (float(box[k]) for k in ("cx", "cy", "w", "h"))
        x1, y1 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
        x2, y2 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 80), width=line_width)
        draw.text((x1, max(0, y1 - 12)), f"GT {int(box['class_id'])}", fill=(0, 255, 80), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def select_dataset_a(root: Path, seed: int, random_per_split: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        images_dir = root / "datasets" / "detection" / "drone_swarm_yolo" / "images" / split
        labels_dir = root / "datasets" / "detection" / "drone_swarm_yolo" / "labels" / split
        images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        records = []
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            boxes = parse_yolo(label)
            areas = [float(b["w"]) * float(b["h"]) for b in boxes if float(b["w"]) > 0 and float(b["h"]) > 0]
            records.append({"split": split, "image": image, "label": label, "boxes": boxes,
                            "median_area": median(areas) if areas else float("inf")})
        if not records:
            continue
        for rec in rng.sample(records, min(random_per_split, len(records))):
            selected[f"{split}:{rec['image'].name}"] = {**rec, "selection": "seeded random"}
        for rec, reason in ((max(records, key=lambda r: len(r["boxes"])), "most crowded in split"),
                            (min(records, key=lambda r: r["median_area"]), "smallest median GT box in split")):
            key = f"{split}:{rec['image'].name}"
            if key in selected:
                selected[key]["selection"] += f"; {reason}"
            else:
                selected[key] = {**rec, "selection": reason}
    return sorted(selected.values(), key=lambda r: (r["split"], r["image"].name.lower()))


def select_dataset_b(root: Path, seed: int, random_per_split: int) -> list[dict[str, Any]]:
    rng = random.Random(seed ^ 0xBEEF)
    selected: dict[str, dict[str, Any]] = {}
    base = root / "datasets" / "UAVSwarm-dataset-master" / "UAVSwarm-dataset-master"
    for split in ("train", "test"):
        split_dir = base / split
        sequences = sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
        picks = rng.sample(sequences, min(random_per_split, len(sequences)))
        # Include a coordinate edge case when supplied MOT rows include a box
        # wholly beyond the frame. This tests candidate predictions against a
        # knownly-suspicious GT frame without trusting that annotation.
        edge_candidates: list[tuple[Path, int, int]] = []
        for seq in sequences:
            meta: dict[str, int] = {}
            ini = seq / "seqinfo.ini"
            if ini.exists():
                for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() in {"imWidth", "imHeight"}:
                            try:
                                meta[k.strip()] = int(v.strip())
                            except ValueError:
                                pass
            gt = seq / "gt" / "gt.txt"
            if not gt.exists() or len(meta) != 2:
                continue
            with gt.open(encoding="utf-8-sig", errors="replace") as stream:
                for line in stream:
                    fields = line.replace(",", " ").split()
                    if len(fields) < 6:
                        continue
                    try:
                        frame = int(float(fields[0]))
                        x, y, w, h = map(float, fields[2:6])
                    except ValueError:
                        continue
                    if w > 0 and h > 0 and (x + w <= 0 or y + h <= 0 or x >= meta["imWidth"] or y >= meta["imHeight"]):
                        edge_candidates.append((seq, frame, 1))
        if edge_candidates:
            edge_candidates.sort(key=lambda x: (x[0].name.lower(), x[1]))
            seq, frame, _ = edge_candidates[len(edge_candidates) // 2]
            if seq not in picks:
                picks.append(seq)
            edge_seq, edge_frame = seq, frame
        else:
            edge_seq = None
            edge_frame = None
        for seq in picks:
            frames = sorted(p for p in (seq / "img1").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not frames:
                continue
            frame = next((p for p in frames if p.stem.isdigit() and int(p.stem) == edge_frame), None) if seq == edge_seq else None
            if frame is None:
                frame = frames[rng.randrange(len(frames))]
            selected[f"{split}:{seq.name}:{frame.name}"] = {"split": split, "sequence": seq.name, "image": frame,
                                                              "selection": "out-of-frame annotation example" if seq == edge_seq and frame.stem == str(edge_frame) else "seeded random sequence/frame"}
    return sorted(selected.values(), key=lambda r: (r["split"], r["sequence"], r["image"].name))


def copy_validation_split(root: Path, out: Path) -> Path:
    source = root / "datasets" / "detection" / "drone_swarm_yolo"
    target = out / "validation_data"
    for subdir in ("images/val", "labels/val"):
        (target / subdir).mkdir(parents=True, exist_ok=True)
    for path in sorted((source / "images/val").iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(path, target / "images/val" / path.name)
    for path in sorted((source / "labels/val").glob("*.txt")):
        shutil.copy2(path, target / "labels/val" / path.name)
    yaml_path = target / "data.yaml"
    yaml_path.write_text(
        f"path: {target.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n  0: drone\n",
        encoding="utf-8",
    )
    return yaml_path


def draw_mot(image_path: Path, seq_dir: Path, output: Path) -> int:
    """Draw supplied MOT rows for this frame; output stays explicitly untrusted."""
    from PIL import Image, ImageDraw, ImageFont

    frame_no = int(image_path.stem)
    width, height = Image.open(image_path).size
    boxes: list[tuple[int, float, float, float, float]] = []
    gt_path = seq_dir / "gt" / "gt.txt"
    if gt_path.exists():
        with gt_path.open(encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                fields = line.replace(",", " ").split()
                if len(fields) < 6:
                    continue
                try:
                    frame, track = int(float(fields[0])), int(float(fields[1]))
                    x, y, w, h = map(float, fields[2:6])
                except ValueError:
                    continue
                if frame == frame_no:
                    boxes.append((track, x, y, w, h))
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for track, x, y, w, h in boxes:
        x2, y2 = x + w, y + h
        inside = x >= 0 and y >= 0 and x2 <= width and y2 <= height
        disjoint = x2 <= 0 or y2 <= 0 or x >= width or y >= height
        state = "inside" if inside else "outside" if disjoint else "partial"
        color = {"inside": (0, 255, 80), "partial": (255, 190, 0), "outside": (255, 50, 50)}[state]
        x1i, y1i, x2i, y2i = map(round, (x, y, x2, y2))
        draw.rectangle((x1i, y1i, x2i, y2i), outline=color, width=max(2, min(width, height) // 400))
        draw.text((x1i, max(0, y1i - 12)), f"SUPPLIED ID {track} {state}", fill=color, font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return len(boxes)


def gallery_page(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        meta = " · ".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in row.items()
                          if k not in {"gt_overlay", "prediction_overlay"})
        left = f'<img src="{html.escape(row["gt_overlay"])}" alt="existing label overlay">' if row.get("gt_overlay") else "<p>No annotation overlay available.</p>"
        right = f'<img src="{html.escape(row["prediction_overlay"])}" alt="checkpoint predictions">'
        cards.append(f"<article><h2>{meta}</h2><div class='pair'><figure><figcaption>GT / supplied annotation</figcaption>{left}</figure><figure><figcaption>Checkpoint prediction</figcaption>{right}</figure></div></article>")
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>
body{{font:15px/1.45 system-ui,sans-serif;margin:1.5rem;background:#f4f6f8;color:#17202a}}.note{{background:#fff3cd;padding:1rem}}article{{background:white;margin:1rem 0;padding:1rem}}.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}figure{{margin:0}}img{{width:100%;height:auto;display:block}}figcaption{{font-weight:600;padding:.3rem 0}}
</style></head><body><h1>{html.escape(title)}</h1><p class="note">Ground-truth or supplied annotation is displayed only for comparison. Model predictions are candidates, not labels. Predictions use the stated confidence threshold and the local checkpoint only.</p>{''.join(cards)}</body></html>""", encoding="utf-8")


def run_predictions(model: Any, rows: list[dict[str, Any]], kind: str, root: Path, out: Path,
                    conf: float, imgsz: int, device: str) -> list[dict[str, Any]]:
    import cv2

    records = []
    for row in rows:
        image_path: Path = row["image"]
        prediction = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, device=device,
                                   verbose=False, save=False)[0]
        pred_rel = Path(kind) / "predictions" / f"{row['split']}_{image_path.stem}_pred.png"
        pred_path = out / pred_rel
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        plotted = prediction.plot(labels=True, conf=True)
        if not cv2.imwrite(str(pred_path), plotted):
            raise OSError(f"Could not write prediction image: {pred_path}")
        boxes = []
        if prediction.boxes is not None:
            xyxy = prediction.boxes.xyxy.cpu().tolist()
            scores = prediction.boxes.conf.cpu().tolist()
            classes = prediction.boxes.cls.cpu().tolist()
            boxes = [{"xyxy": [round(float(v), 2) for v in box], "confidence": round(float(score), 5), "class_id": int(cls)}
                     for box, score, cls in zip(xyxy, scores, classes)]
        record = {k: (str(v.relative_to(root).as_posix()) if isinstance(v, Path) else v)
                  for k, v in row.items() if k not in {"boxes", "label"}}
        record.update({"prediction_count": len(boxes), "predictions": boxes, "confidence_threshold": conf,
                       "imgsz": imgsz, "device": device, "prediction_overlay": pred_rel.as_posix()})
        if kind == "dataset_a":
            gt_rel = Path("dataset_a") / "gt" / f"{row['split']}_{image_path.stem}_gt.png"
            draw_gt(image_path, row["boxes"], out / gt_rel)
            record["gt_overlay"] = gt_rel.as_posix()
            record["gt_box_count"] = len(row["boxes"])
        else:
            gt_rel = Path("dataset_b") / "supplied_annotations" / f"{row['split']}_{row['sequence']}_{image_path.stem}_supplied_mot.png"
            record["supplied_mot_box_count"] = draw_mot(image_path, image_path.parent.parent, out / gt_rel)
            record["gt_overlay"] = gt_rel.as_posix()
            record["annotation_status"] = "supplied MOT; unverified"
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/detect/runs/drone_detection/smoke_test/weights/best.pt"))
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--random-per-split", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the copied validation-set evaluation")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.output.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to write into nonempty output directory: {out}")
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.random_per_split < 0 or not 0 <= args.conf <= 1:
        parser.error("random-per-split must be nonnegative and conf must be in [0, 1]")

    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    a_rows = select_dataset_a(root, args.seed, args.random_per_split)
    b_rows = select_dataset_b(root, args.seed, args.random_per_split)
    a_records = run_predictions(model, a_rows, "dataset_a", root, out, args.conf, args.imgsz, args.device)
    b_records = run_predictions(model, b_rows, "dataset_b", root, out, args.conf, args.imgsz, args.device)
    gallery_page(out / "model_review_dataset_a.html", "Dataset A — GT vs existing checkpoint", a_records)
    gallery_page(out / "model_review_dataset_b.html", "Dataset B — supplied annotations vs existing checkpoint", b_records)

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint), "loaded": True, "seed": args.seed, "conf": args.conf,
        "imgsz": args.imgsz, "device": args.device,
        "dataset_a": a_records, "dataset_b": b_records,
        "notes": ["Dataset A samples include seeded random, most crowded, and smallest median ground-truth box examples per split.",
                  "Dataset B is inference-only; its annotations are not converted into or used as training labels.",
                  "All prediction overlays are generated from the local checkpoint; no weights are downloaded."],
    }

    if not args.skip_validation:
        val_yaml = copy_validation_split(root, out)
        val_result = model.val(data=str(val_yaml), split="val", imgsz=args.imgsz, device=args.device,
                               batch=8, workers=0, plots=False, save_json=False, cache=False,
                               project=str(out / "validation"), name="best_checkpoint_val", exist_ok=False,
                               verbose=False)
        report["validation"] = {"data_yaml": str(val_yaml), "copied_images": len(list((out / "validation_data/images/val").iterdir())),
                                "metrics": {key: float(value) for key, value in val_result.results_dict.items()}}
    (out / "model_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checkpoint loaded: {checkpoint}")
    print(f"Dataset A comparison gallery: {out / 'model_review_dataset_a.html'}")
    print(f"Dataset B comparison gallery: {out / 'model_review_dataset_b.html'}")
    if "validation" in report:
        print(f"Validation metrics: {json.dumps(report['validation']['metrics'], sort_keys=True)}")
    else:
        print("Validation metrics: skipped by request")
    print(f"Report: {out / 'model_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
