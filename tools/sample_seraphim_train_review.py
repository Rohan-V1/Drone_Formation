"""Create a reproducible visual sample of Seraphim train images and boxes."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "datasets/external_seraphim_detector_v1/prepared/extracted/train"
DEFAULT_OUTPUT = PROJECT / "outputs/seraphim_detector_build/run_001"
SEED = 42


def read_boxes(label: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in label.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) != 5:
            raise ValueError(f"Expected staged YOLO box row in {label}: {line}")
        _, cx, cy, w, h = vals
        rows.append((float(cx), float(cy), float(w), float(h)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-count", type=int, default=24)
    parser.add_argument("--small-count", type=int, default=8)
    parser.add_argument("--multi-count", type=int, default=8)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite review output: {out}")
    images_dir, labels_dir = data_root / "images", data_root / "labels"
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No staged Seraphim images under {images_dir}")

    rng = random.Random(SEED)
    pool = rng.sample(images, min(4096, len(images)))
    box_map = {}
    areas = {}
    multi = []
    for image in pool:
        boxes = read_boxes(labels_dir / f"{image.stem}.txt")
        box_map[image] = boxes
        img_area = 1.0
        areas[image] = min((w * h / img_area for _, _, w, h in boxes), default=1.0)
        if len(boxes) >= 2:
            multi.append(image)
    random_sample = rng.sample(pool, min(args.random_count, len(pool)))
    tiny_pool = sorted((p for p in pool if box_map[p]), key=lambda p: (areas[p], p.name))
    tiny_sample = tiny_pool[:min(args.small_count, len(tiny_pool))]
    multi_sample = rng.sample(multi, min(args.multi_count, len(multi))) if multi else []
    groups = [("random", random_sample), ("smallest_box", tiny_sample), ("multi_object", multi_sample)]

    out.mkdir(parents=True, exist_ok=False)
    cell_w, cell_h = 310, 285
    cols = 4
    font = ImageFont.load_default()
    records = []
    for group, selected in groups:
        rows = (len(selected) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, max(1, rows) * cell_h), "#111827")
        canvas = ImageDraw.Draw(sheet)
        for idx, image_path in enumerate(selected):
            boxes = box_map[image_path]
            with Image.open(image_path) as im:
                original = im.convert("RGB")
            draw = ImageDraw.Draw(original)
            width, height = original.size
            for box_idx, (cx, cy, bw, bh) in enumerate(boxes, 1):
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                draw.rectangle((x1, y1, x2, y2), outline=(20, 255, 100), width=2)
                draw.text((x1, max(0, y1 - 12)), f"0:{box_idx}", fill=(0, 0, 0), stroke_width=1, stroke_fill=(20, 255, 100))
            thumb = original.copy()
            thumb.thumbnail((cell_w - 20, cell_h - 52), Image.Resampling.LANCZOS)
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            ox = x + (cell_w - thumb.width) // 2
            oy = y + 5
            sheet.paste(thumb, (ox, oy))
            canvas.text((x + 8, y + cell_h - 38), f"{image_path.name} | {len(boxes)} boxes | min area {areas[image_path]:.5f}", font=font, fill="white")
            records.append({
                "sample_group": group,
                "image": str(image_path),
                "image_name": image_path.name,
                "width": width,
                "height": height,
                "box_count": len(boxes),
                "minimum_box_area_fraction": areas[image_path],
                "boxes_yolo": [list(b) for b in boxes],
            })
        sheet_path = out / f"{group}_contact_sheet.jpg"
        sheet.save(sheet_path, quality=92)

    with (out / "sample_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_group", "image_name", "width", "height", "box_count", "minimum_box_area_fraction", "boxes_yolo"])
        writer.writeheader()
        for row in records:
            writer.writerow({
                "sample_group": row["sample_group"],
                "image_name": row["image_name"],
                "width": row["width"],
                "height": row["height"],
                "box_count": row["box_count"],
                "minimum_box_area_fraction": row["minimum_box_area_fraction"],
                "boxes_yolo": json.dumps(row["boxes_yolo"]),
            })
    report = {
        "source": "Seraphim official training split, prepared derived copy",
        "random_seed": SEED,
        "image_count_available": len(images),
        "sample_count": len(records),
        "sample_groups": {name: len(items) for name, items in groups},
        "selection_pool_size": len(pool),
        "images_with_multiple_boxes_in_selection_pool": len(multi),
        "smallest_boxes_are_ranked_within_seeded_selection_pool_only": True,
        "outputs": [f"{name}_contact_sheet.jpg" for name, _ in groups] + ["sample_records.csv"],
        "interpretation_limit": "This is a visual spot-check, not proof of dataset-wide label correctness or real-only provenance.",
    }
    (out / "sample_review_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
