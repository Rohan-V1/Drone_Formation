"""Local-only manual annotation server for the UAVSwarm pilot set.

The tool intentionally reads only manifest-listed images. It never reads the
dataset's supplied annotations or model predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "UAVSwarm-dataset-master" / "UAVSwarm-dataset-master" / "train"
DATA_ROOT = PROJECT_ROOT / "datasets" / "derived_swarm_labels_v1"
UI_FILE = Path(__file__).resolve().parent / "annotation_ui.html"
TRAIN_SEQUENCES = ["UAVSwarm-05", "UAVSwarm-11", "UAVSwarm-19", "UAVSwarm-27", "UAVSwarm-33", "UAVSwarm-39", "UAVSwarm-47", "UAVSwarm-65", "UAVSwarm-69"]
VAL_SEQUENCES = ["UAVSwarm-07", "UAVSwarm-29", "UAVSwarm-67"]
SEQUENCE_RATIONALE = {
    "UAVSwarm-05": "cloudy sky with a multi-drone formation; 446x270 source frames",
    "UAVSwarm-11": "cloudy sky with many tiny/distant targets",
    "UAVSwarm-19": "wooded landscape background",
    "UAVSwarm-27": "built/urban background and distant targets",
    "UAVSwarm-33": "bright cyan sky and several separated drones",
    "UAVSwarm-39": "high-resolution 1919x1079 landscape/horizon scene",
    "UAVSwarm-47": "multi-drone sky with buildings/trees in the background",
    "UAVSwarm-65": "two comparatively close and visually distinct drones",
    "UAVSwarm-69": "park/field scene with distant targets",
    "UAVSwarm-07": "cloudy sky and formation, held out by sequence",
    "UAVSwarm-29": "park/open-field background, held out by sequence",
    "UAVSwarm-67": "low viewpoint with a close foreground drone, held out by sequence",
}

app = FastAPI(title="Drone manual annotation pilot", docs_url=None, redoc_url=None)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def selected_indices(count: int, sample_count: int = 10) -> list[int]:
    """Deterministic midpoint samples across a sequence, zero-based."""
    if count < sample_count:
        raise ValueError(f"Need at least {sample_count} frames, found {count}")
    return [min(count - 1, int((i + 0.5) * count / sample_count)) for i in range(sample_count)]


def make_manifest() -> dict:
    if set(TRAIN_SEQUENCES) & set(VAL_SEQUENCES):
        raise RuntimeError("Sequence leakage between train and validation")
    images: list[dict] = []
    for split, sequences in (("train", TRAIN_SEQUENCES), ("val", VAL_SEQUENCES)):
        for sequence in sequences:
            seq_dir = SOURCE_ROOT / sequence
            if not seq_dir.is_dir():
                raise FileNotFoundError(f"Selected train sequence missing: {seq_dir}")
            files = sorted((seq_dir / "img1").glob("*.jpg"), key=lambda p: int(p.stem))
            indices = selected_indices(len(files))
            for idx in indices:
                path = files[idx]
                # Pillow is already part of the project's existing environment.
                from PIL import Image
                with Image.open(path) as im:
                    width, height = im.size
                    im.verify()
                frame_number = int(path.stem)
                image_id = f"{sequence}_{frame_number:06d}"
                images.append({
                    "image_id": image_id,
                    "sequence_id": sequence,
                    "frame_number": frame_number,
                    "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "split": split,
                    "width": width,
                    "height": height,
                    "selection_index_0based": idx,
                    "sequence_frame_count": len(files),
                    "selection_rationale": SEQUENCE_RATIONALE[sequence],
                })
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "dataset": "Dataset B train only; no supplied labels or predictions imported",
        "selection": {
            "algorithm": "For each selected sequence, sort img1/*.jpg numerically and select indices floor((i+0.5)*N/10), i=0..9.",
            "seed": None,
            "frames_per_sequence": 10,
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VAL_SEQUENCES,
            "official_test_sequences_used": False,
        },
        "class_names": ["drone"],
        "images": images,
    }


def manifest_path() -> Path:
    return DATA_ROOT / "manifest.json"


def load_manifest() -> dict:
    path = manifest_path()
    if not path.exists():
        raise HTTPException(status_code=500, detail="Pilot manifest has not been initialized")
    return json.loads(path.read_text(encoding="utf-8"))


def annotation_path(item: dict) -> Path:
    return DATA_ROOT / "annotations" / item["split"] / f"{item['image_id']}.json"


def label_path(item: dict) -> Path:
    return DATA_ROOT / "labels" / item["split"] / f"{item['image_id']}.txt"


def get_item(image_id: str) -> dict:
    for item in load_manifest()["images"]:
        if item["image_id"] == image_id:
            return item
    raise HTTPException(status_code=404, detail="Image is not part of the pilot manifest")


def read_annotation(item: dict) -> dict:
    p = annotation_path(item)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"image_id": item["image_id"], "status": "unreviewed", "boxes": [], "updated_utc": None}


def write_progress() -> dict:
    manifest = load_manifest()
    counts = {s: {"unreviewed": 0, "draft": 0, "reviewed": 0} for s in ("train", "val")}
    completed = 0
    for item in manifest["images"]:
        status = read_annotation(item)["status"]
        counts[item["split"]][status] += 1
        completed += status == "reviewed"
    progress = {"updated_utc": utc_now(), "total": len(manifest["images"]), "reviewed": completed, "by_split": counts}
    atomic_json(DATA_ROOT / "progress.json", progress)
    return progress


def write_validation_report() -> dict:
    manifest = load_manifest()
    split_sequences = {s: {x["sequence_id"] for x in manifest["images"] if x["split"] == s} for s in ("train", "val")}
    image_count = {s: 0 for s in ("train", "val")}
    reviewed_count = {s: 0 for s in ("train", "val")}
    errors: list[str] = []
    for item in manifest["images"]:
        image_count[item["split"]] += 1
        p = annotation_path(item)
        if not p.exists():
            errors.append(f"{item['image_id']}: annotation JSON missing")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("status") not in ("unreviewed", "draft", "reviewed"):
            errors.append(f"{item['image_id']}: invalid annotation status")
        if data.get("status") == "reviewed":
            reviewed_count[item["split"]] += 1
        boxes = data.get("boxes", [])
        for n, b in enumerate(boxes):
            vals = [b.get(k) for k in ("x1", "y1", "x2", "y2")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                errors.append(f"{item['image_id']} box {n}: non-finite/non-numeric coordinate")
            elif not (0 <= vals[0] < vals[2] <= item["width"] and 0 <= vals[1] < vals[3] <= item["height"]):
                errors.append(f"{item['image_id']} box {n}: out-of-bounds or zero-area")
        lp = label_path(item)
        if data.get("status") == "reviewed":
            if not lp.exists():
                errors.append(f"{item['image_id']}: reviewed image has no YOLO label file")
            else:
                lines = lp.read_text(encoding="utf-8").splitlines()
                if len(lines) != len(boxes):
                    errors.append(f"{item['image_id']}: {len(boxes)} JSON boxes vs {len(lines)} YOLO rows")
                for n, line in enumerate(lines, 1):
                    parts = line.split()
                    try:
                        nums = [float(v) for v in parts[1:]]
                        if len(parts) != 5 or parts[0] != "0" or len(nums) != 4 or not all(math.isfinite(v) for v in nums):
                            raise ValueError()
                        cx, cy, w, h = nums
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1 and cx - w/2 >= -1e-8 and cy - h/2 >= -1e-8 and cx + w/2 <= 1+1e-8 and cy + h/2 <= 1+1e-8):
                            raise ValueError()
                    except Exception:
                        errors.append(f"{item['image_id']} YOLO row {n}: invalid class/coordinates")
        elif lp.exists():
            errors.append(f"{item['image_id']}: {data['status']} image has a YOLO label file; expected labels only for reviewed images")
    overlap = sorted(split_sequences["train"] & split_sequences["val"])
    report = {
        "updated_utc": utc_now(),
        "image_to_annotation_json_correspondence": {"expected": len(manifest["images"]), "found": sum(annotation_path(x).exists() for x in manifest["images"])},
        "images_by_split": image_count,
        "reviewed_by_split": reviewed_count,
        "train_validation_sequence_overlap": overlap,
        "coordinate_and_yolo_checks": "all box coordinates finite, positive area, in image bounds; reviewed rows class 0 and normalized/in bounds",
        "errors": errors,
        "valid": not errors and not overlap,
        "human_visual_correctness_verified": False,
    }
    atomic_json(DATA_ROOT / "validation_report.json", report)
    return report


def ensure_data() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    mp = manifest_path()
    if not mp.exists():
        manifest = make_manifest()
        if len(manifest["images"]) != 120:
            raise RuntimeError(f"Expected 120 pilot images, selected {len(manifest['images'])}")
        atomic_json(mp, manifest)
    manifest = load_manifest()
    for item in manifest["images"]:
        p = annotation_path(item)
        if not p.exists():
            atomic_json(p, {"image_id": item["image_id"], "status": "unreviewed", "boxes": [], "updated_utc": None})
    write_progress()
    write_validation_report()


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SaveBody(BaseModel):
    status: Literal["draft", "reviewed"]
    boxes: list[Box] = Field(default_factory=list)


@app.get("/", response_class=HTMLResponse)
def home():
    ensure_data()
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.get("/api/manifest")
def api_manifest():
    ensure_data()
    return load_manifest()


@app.get("/api/progress")
def api_progress():
    ensure_data()
    return write_progress()


@app.get("/api/annotation/{image_id}")
def api_annotation(image_id: str):
    ensure_data()
    item = get_item(image_id)
    return read_annotation(item)


@app.get("/api/image/{image_id}")
def api_image(image_id: str):
    ensure_data()
    item = get_item(image_id)
    path = (PROJECT_ROOT / item["source_path"]).resolve()
    if not path.is_file() or not path.is_relative_to(SOURCE_ROOT.resolve()):
        raise HTTPException(status_code=404, detail="Manifest image is unavailable")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/annotation/{image_id}")
def api_save(image_id: str, body: SaveBody):
    ensure_data()
    item = get_item(image_id)
    clean = []
    for n, box in enumerate(body.boxes, 1):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise HTTPException(422, f"Box {n} has a non-finite coordinate. Nothing was saved.")
        if not (0 <= x1 < x2 <= item["width"] and 0 <= y1 < y2 <= item["height"]):
            raise HTTPException(422, f"Box {n} must have positive area and lie fully inside the {item['width']}x{item['height']} image. Nothing was clipped or discarded; adjust it and save again.")
        clean.append({"class_id": 0, "class_name": "drone", "x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)})
    data = {"image_id": image_id, "sequence_id": item["sequence_id"], "frame_number": item["frame_number"], "split": item["split"], "image_width": item["width"], "image_height": item["height"], "status": body.status, "boxes": clean, "updated_utc": utc_now()}
    atomic_json(annotation_path(item), data)
    lp = label_path(item)
    if body.status == "reviewed":
        lp.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for b in clean:
            bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
            cx, cy = (b["x1"] + b["x2"]) / 2 / item["width"], (b["y1"] + b["y2"]) / 2 / item["height"]
            rows.append(f"0 {cx:.8f} {cy:.8f} {bw/item['width']:.8f} {bh/item['height']:.8f}")
        lp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n")
    elif lp.exists():
        # A draft is not training data; remove the generated label from a previously reviewed item.
        lp.unlink()
    progress = write_progress()
    report = write_validation_report()
    return {"saved": True, "status": body.status, "box_count": len(clean), "progress": progress, "validation_valid": report["valid"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ensure_data()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
