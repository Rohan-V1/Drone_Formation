"""Local review-only UI for the Phase 3D flagged UAVSwarm annotation cases."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "datasets/derived_swarm_labels_v1"
SOURCE_TRAIN = (ROOT / "datasets/UAVSwarm-dataset-master/UAVSwarm-dataset-master/train").resolve()
REVIEW_DIR = ROOT / "outputs/phase3d_adjudication_001"
REGISTRY_PATH = REVIEW_DIR / "case_registry.json"
CHANGE_LOG = PILOT / "annotation_change_log.jsonl"
POLICY_VERSION = "1.0"
CASES = {
    "UAVSwarm-29_000042": [1, 2],
    "UAVSwarm-29_000059": [1, 2, 3],
    "UAVSwarm-29_000092": [6, 7],
    "UAVSwarm-29_000125": [2, 3],
    "UAVSwarm-27_000010": [1],
}
LOCK = threading.Lock()
app = FastAPI(title="Phase 3D annotation adjudication", docs_url=None, redoc_url=None)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def manifest() -> dict:
    p = PILOT / "manifest.json"
    if not p.is_file():
        raise RuntimeError(f"Pilot manifest missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def items_by_id() -> dict[str, dict]:
    return {x["image_id"]: x for x in manifest()["images"]}


def annotation_file(item: dict) -> Path:
    return PILOT / "annotations" / item["split"] / f"{item['image_id']}.json"


def label_file(item: dict) -> Path:
    return PILOT / "labels" / item["split"] / f"{item['image_id']}.txt"


def make_registry() -> dict:
    indexed = items_by_id()
    registry_cases = []
    for image_id, box_indices in CASES.items():
        item = indexed.get(image_id)
        if not item:
            raise RuntimeError(f"Flagged image absent from manifest: {image_id}")
        p = annotation_file(item)
        data = json.loads(p.read_text(encoding="utf-8"))
        boxes = data.get("boxes", [])
        for box_index in box_indices:
            if box_index < 1 or box_index > len(boxes):
                raise RuntimeError(f"{image_id}: expected box {box_index}, found {len(boxes)} boxes")
            b = boxes[box_index - 1]
            coords = {k: float(b[k]) for k in ("x1", "y1", "x2", "y2")}
            if not all(math.isfinite(v) for v in coords.values()):
                raise RuntimeError(f"{image_id} box {box_index}: non-finite coordinates")
            registry_cases.append({
                "case_id": f"{image_id}#{box_index}", "image_id": image_id,
                "box_index_1based": box_index, "original_box": b,
                "original_coordinates": coords, "source_path": item["source_path"],
                "split": item["split"], "sequence_id": item["sequence_id"],
                "frame_number": item["frame_number"], "width": item["width"],
                "height": item["height"],
            })
    return {"schema_version": 1, "created_utc": now(), "policy_version": POLICY_VERSION, "cases": registry_cases}


def registry() -> dict:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        # Exclusive create makes a repeat launch safe and never overwrites a prior snapshot.
        data = make_registry()
        try:
            with REGISTRY_PATH.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
        except FileExistsError:
            pass
    result = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    expected = {f"{image_id}#{i}" for image_id, indices in CASES.items() for i in indices}
    actual = {x["case_id"] for x in result.get("cases", [])}
    if actual != expected:
        raise RuntimeError("Saved review case registry does not match the flagged case list; refusing to rewrite it")
    return result


def read_log() -> list[dict]:
    if not CHANGE_LOG.exists():
        return []
    events = []
    for n, line in enumerate(CHANGE_LOG.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Change log line {n} is invalid JSON: {exc}") from exc
    return events


def append_log(event: dict) -> None:
    CHANGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CHANGE_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def decision_events() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    applied: dict[str, bool] = {}
    for event in read_log():
        if event.get("event_type") == "adjudication" and event.get("case_id"):
            latest[event["case_id"]] = event
            applied[event["record_id"]] = bool(event.get("no_data_change"))
        elif event.get("event_type") == "application":
            applied[event.get("record_id", "")] = bool(event.get("applied"))
    for case_id, event in latest.items():
        event["application_complete"] = applied.get(event["record_id"], False)
    return latest


def same_coords(a: dict, b: dict, tol: float = 0.00011) -> bool:
    try:
        return all(abs(float(a[k]) - float(b[k])) <= tol for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False


def box_present(boxes: list[dict], case: dict) -> bool:
    return any(same_coords(b, case["original_coordinates"]) for b in boxes)


def read_cases() -> list[dict]:
    saved = registry()
    indexed = items_by_id()
    latest = decision_events()
    out = []
    for case in saved["cases"]:
        item = indexed[case["image_id"]]
        current = json.loads(annotation_file(item).read_text(encoding="utf-8"))
        decision = latest.get(case["case_id"])
        out.append({
            **case,
            "image_url": f"/api/image/{case['image_id']}",
            "target_present": box_present(current.get("boxes", []), case),
            "other_boxes": [b for b in current.get("boxes", []) if not same_coords(b, case["original_coordinates"])],
            "decision": decision.get("decision") if decision else None,
            "reason": decision.get("reason", "") if decision else "",
            "application_complete": decision.get("application_complete") if decision else None,
            "annotation_status": current.get("status"),
        })
    return out


@app.get("/", response_class=HTMLResponse)
def home():
    ui = (Path(__file__).with_name("phase3d_adjudication_ui.html")).read_text(encoding="utf-8")
    return HTMLResponse(ui)


@app.get("/api/cases")
def api_cases():
    cases = read_cases()
    counts = {k: sum(c["decision"] == k for c in cases) for k in ("keep", "remove", "uncertain")}
    counts["pending"] = sum(c["decision"] is None for c in cases)
    return {"cases": cases, "counts": counts, "change_log": str(CHANGE_LOG.resolve()), "policy_version": POLICY_VERSION}


@app.get("/api/image/{image_id}")
def api_image(image_id: str):
    item = items_by_id().get(image_id)
    if not item:
        raise HTTPException(404, "Image is not in the pilot manifest")
    path = (ROOT / item["source_path"]).resolve()
    if not path.is_file() or not path.is_relative_to(SOURCE_TRAIN):
        raise HTTPException(404, "Source image is unavailable or outside Dataset B train")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class DecisionBody(BaseModel):
    decision: str
    reason: str
    confirm_remove: bool = False


@app.post("/api/decision/{case_id:path}")
def save_decision(case_id: str, body: DecisionBody):
    if body.decision not in ("keep", "remove", "uncertain"):
        raise HTTPException(422, "Decision must be keep, remove, or uncertain")
    reason = body.reason.strip()
    if len(reason) < 3:
        raise HTTPException(422, "Enter a brief reason before recording this decision")
    if body.decision == "remove" and not body.confirm_remove:
        raise HTTPException(422, "Removal requires explicit confirmation")
    with LOCK:
        case = next((c for c in registry()["cases"] if c["case_id"] == case_id), None)
        if not case:
            raise HTTPException(404, "Adjudication case not found")
        indexed = items_by_id()
        item = indexed[case["image_id"]]
        ap = annotation_file(item)
        lp = label_file(item)
        data = json.loads(ap.read_text(encoding="utf-8"))
        boxes = list(data.get("boxes", []))
        positions = [i for i, b in enumerate(boxes) if same_coords(b, case["original_coordinates"])]
        if len(positions) > 1:
            raise HTTPException(409, "More than one box matches the flagged coordinates; no change made")
        present = bool(positions)
        previous_decision = decision_events().get(case_id)
        previous_coordinates = case["original_coordinates"]
        mutation = "none"
        if body.decision == "remove" and present:
            boxes.pop(positions[0])
            mutation = "removed"
        elif body.decision == "remove" and not present:
            if not previous_decision or previous_decision.get("decision") != "remove" or not previous_decision.get("application_complete"):
                raise HTTPException(409, "Flagged box is already absent without a completed prior removal record; no change made")
        elif body.decision == "keep" and not present:
            if not previous_decision or previous_decision.get("decision") != "remove" or not previous_decision.get("application_complete"):
                raise HTTPException(409, "Flagged box is absent without a completed prior removal record; no change made")
            restore_at = min(max(case["box_index_1based"] - 1, 0), len(boxes))
            boxes.insert(restore_at, case["original_box"])
            mutation = "restored"
        record_id = str(uuid.uuid4())
        event = {
            "event_type": "adjudication", "record_id": record_id, "timestamp_utc": now(),
            "policy_version": POLICY_VERSION, "image_id": case["image_id"],
            "box_index_1based": case["box_index_1based"], "case_id": case_id,
            "previous_coordinates": previous_coordinates, "decision": body.decision,
            "reason": reason, "annotation_action": mutation,
            "no_data_change": mutation == "none", "application_pending": mutation != "none",
        }
        append_log(event)  # Write-ahead decision record; failures remain visible in the UI/audit log.
        if mutation != "none":
            data["boxes"] = boxes
            data["updated_utc"] = now()
            w, h = int(item["width"]), int(item["height"])
            lines = []
            for b in boxes:
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise HTTPException(409, "Existing annotation has invalid geometry; decision recorded but no data rewrite attempted")
                bw, bh = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
                lines.append(f"0 {cx:.8f} {cy:.8f} {bw/w:.8f} {bh/h:.8f}")
            atomic_json(ap, data)
            atomic_text(lp, "\n".join(lines) + ("\n" if lines else ""))
            append_log({"event_type": "application", "record_id": record_id, "timestamp_utc": now(), "applied": True, "annotation_path": str(ap.resolve()), "label_path": str(lp.resolve())})
        else:
            append_log({"event_type": "application", "record_id": record_id, "timestamp_utc": now(), "applied": True, "changed": False})
        return {"saved": True, "record_id": record_id, "decision": body.decision, "annotation_action": mutation, "case": next(c for c in read_cases() if c["case_id"] == case_id)}


@app.get("/health")
def health():
    return {"ok": True, "case_count": len(registry()["cases"]), "source": "read-only image access; human decisions required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    registry()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
