"""Streaming real-video inference using the saved detector and geometric rules."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path(r"C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt")
CLASSES = ["V", "Echelon", "Column/File", "Line/Abreast", "Diamond/Box", "Circle"]
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_DUPLICATE_IOU_MIN = 0.50
IMAGE_DUPLICATE_SMALLER_BOX_COVERAGE_MIN = 0.85
IMAGE_DUPLICATE_CENTER_DIAGONAL_RATIO_MAX = 0.20
IMAGE_GEOMETRY_VERSION = "chart_12_image_geometry_v2"
VIDEO_GEOMETRY_VERSION = "camera_compensated_temporal_v1"

app = FastAPI(title="Drone Formation Identification AI Service", version="0.3.0")
_runtime = None
_runtime_lock = threading.Lock()
_inference_lock = threading.RLock()


def _load_runtime():
    """Load only verified local artifacts; avoid writing bytecode into past runs."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        return _load_runtime_locked()


def _load_runtime_locked():
    """Load the verified detector and real-observation geometry rules once."""
    global _runtime
    checkpoint = Path(os.environ.get("DRONE_DETECTOR_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    geometry_path = Path(__file__).with_name("spatial_formation_classifier.py")
    required = [checkpoint, geometry_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Required model artifacts are missing: " + ", ".join(missing))

    import numpy as np
    import torch
    import cv2
    from ultralytics import YOLO

    # The helper is loaded directly from this source file; do not leave pyc files.
    sys.dont_write_bytecode = True
    geometry_spec = importlib.util.spec_from_file_location("real_image_spatial_geometry", geometry_path)
    if geometry_spec is None or geometry_spec.loader is None:
        raise RuntimeError("Could not load the real-image spatial geometry rules")
    geometry_module = importlib.util.module_from_spec(geometry_spec)
    sys.modules[geometry_spec.name] = geometry_module
    geometry_spec.loader.exec_module(geometry_module)

    detector = YOLO(str(checkpoint))
    detector_names = detector.names
    name_map = {int(k): str(v).lower() for k, v in detector_names.items()} if isinstance(detector_names, dict) else {i: str(v).lower() for i, v in enumerate(detector_names)}
    if name_map != {0: "drone"}:
        raise RuntimeError(f"Detector checkpoint is not the expected one-class drone model: {name_map}")
    _runtime = {"np": np, "torch": torch, "cv2": cv2, "YOLO": YOLO,
                "checkpoint": str(checkpoint),
                "detector": detector, "geometry_classify": geometry_module.classify,
                "geometry_classify_image": geometry_module.classify_image,
                "geometry_classify_video": geometry_module.classify_video_window,
                "device": "0" if torch.cuda.is_available() else "cpu",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return _runtime


@app.get("/health")
def health_check():
    try:
        import torch
        return {"status": "ok", "service": "drone-formation-ai",
                "image_geometry_version": IMAGE_GEOMETRY_VERSION,
                "python_executable": sys.executable,
                "formation_method": "real-image spatial and temporal geometry rules",
                "video_geometry_version": VIDEO_GEOMETRY_VERSION,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        return {"status": "degraded", "service": "drone-formation-ai", "error": str(exc)}


@app.post("/api/analyze")
async def analyze_video(request: Request):
    """Accept a raw streamed video body to avoid requiring multipart dependencies."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if not (content_type.startswith("video/") or content_type == "application/octet-stream"):
        raise HTTPException(415, "Upload a video file (video/* content type required).")
    raw_name = request.headers.get("x-file-name", "upload.mp4")
    try:
        filename = Path(__import__("urllib.parse", fromlist=["unquote"]).unquote(raw_name)).name
    except Exception:
        filename = "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Supported video extensions: " + ", ".join(sorted(ALLOWED_EXTENSIONS)))

    jobs_root = ROOT / "outputs" / "real_video_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    job_dir = jobs_root / job_id
    job_dir.mkdir(exist_ok=False)
    input_path = job_dir / ("source" + suffix)
    received = 0
    try:
        with input_path.open("xb") as target:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the 1 GiB upload limit.")
                target.write(chunk)
        if received == 0:
            raise HTTPException(400, "The uploaded video is empty.")
        result = await run_in_threadpool(_process_video_serialized, input_path, job_dir, filename, job_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        (job_dir / "error.json").write_text(json.dumps({"job_id": job_id, "error": str(exc)}, indent=2), encoding="utf-8")
        raise HTTPException(422, f"Video processing failed: {exc}") from exc
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/analyze-image")
async def analyze_image(request: Request):
    """Run YOLO image detection and conservative spatial assessment; no tracking."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if not content_type.startswith("image/") and content_type != "application/octet-stream":
        raise HTTPException(415, "Upload an image file (image/* content type required).")
    raw_name = request.headers.get("x-file-name", "upload.jpg")
    from urllib.parse import unquote
    filename = Path(unquote(raw_name)).name
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(415, "Supported image extensions: " + ", ".join(sorted(IMAGE_EXTENSIONS)))

    jobs_root = ROOT / "outputs" / "real_image_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    job_dir = jobs_root / job_id
    job_dir.mkdir(exist_ok=False)
    source_path = job_dir / ("source" + suffix)
    received = 0
    try:
        with source_path.open("xb") as target:
            async for chunk in request.stream():
                received += len(chunk)
                if received > 100 * 1024 * 1024:
                    raise HTTPException(413, "Image exceeds the 100 MiB upload limit.")
                target.write(chunk)
        if received == 0:
            raise HTTPException(400, "The uploaded image is empty.")
        result = await run_in_threadpool(_process_image_serialized, source_path, job_dir, filename, job_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        (job_dir / "error.json").write_text(json.dumps({"job_id": job_id, "error": str(exc)}, indent=2), encoding="utf-8")
        raise HTTPException(422, f"Image processing failed: {exc}") from exc
    finally:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            pass


def _process_image_serialized(source_path: Path, job_dir: Path, filename: str, job_id: str):
    with _inference_lock:
        return _process_image(source_path, job_dir, filename, job_id)


def _process_image(source_path: Path, job_dir: Path, filename: str, job_id: str):
    rt = _load_runtime()
    cv2 = rt["cv2"]
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode this image.")
    height, width = image.shape[:2]
    detector = rt["detector"]
    # Keep the previously used Ultralytics defaults explicit: confidence 0.15,
    # class-aware NMS at IoU 0.70. The stricter still-image cleanup below is a
    # separate geometric pass and does not alter raw detector outputs.
    result = detector.predict(image, conf=0.15, imgsz=640, iou=0.70,
                              agnostic_nms=False, verbose=False, device=rt["device"])[0]
    boxes = result.boxes
    detection_rows = []
    boxes_xyxy = []
    confidences = []
    if boxes is not None:
        for index, (coords, conf, cls) in enumerate(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.int().cpu().tolist())):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in coords]
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            row = {"detector_box_id": index + 1, "image_name": filename, "class_id": 0, "class_name": "drone", "track_id": "",
                   "confidence": round(float(conf), 6), "x1": round(x1, 3), "y1": round(y1, 3),
                   "x2": round(x2, 3), "y2": round(y2, 3), "counted_as_distinct": True,
                   "suppressed_by_detection_id": ""}
            detection_rows.append(row)
            boxes_xyxy.append((x1, y1, x2, y2))
            confidences.append(float(conf))

    kept_indices, duplicate_suppressions = _suppress_duplicate_image_boxes(boxes_xyxy, confidences)
    suppressed_by = {item["detection_id"]: item["representative_detection_id"] for item in duplicate_suppressions}
    for row in detection_rows:
        raw_id = row["detector_box_id"]
        if raw_id in suppressed_by:
            row["counted_as_distinct"] = False
            row["suppressed_by_detection_id"] = suppressed_by[raw_id]

    distinct_rows = [detection_rows[i] for i in kept_indices]
    distinct_boxes = [boxes_xyxy[i] for i in kept_indices]
    box_points = [((box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height))
                  for box in distinct_boxes]

    # Draw only retained boxes. Confidence text over tiny, crowded drones made
    # the image unreadable; exact IDs and confidences remain available in CSV.
    box_line_width = max(1, round(min(width, height) / 500.0))
    box_color = (70, 235, 170)
    for raw_index in kept_indices:
        x1, y1, x2, y2 = boxes_xyxy[raw_index]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), box_color, box_line_width)
    annotated_path = job_dir / "annotated.png"
    if not cv2.imwrite(str(annotated_path), image):
        raise RuntimeError("OpenCV failed to save the annotated image.")
    detections_path = job_dir / "detections.csv"
    fields = ["detector_box_id", "image_name", "class_id", "class_name", "track_id", "confidence",
              "x1", "y1", "x2", "y2", "counted_as_distinct", "suppressed_by_detection_id"]
    with detections_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(detection_rows)
    formation = _assess_image_formation(box_points, distinct_boxes, rt["geometry_classify_image"],
                                        [row["confidence"] for row in distinct_rows])
    unresolved_overlap_pairs = formation.pop("overlapping_detection_pairs")

    metadata = {"job_id": job_id, "input_filename": filename, "width": width, "height": height,
                "detector_checkpoint": rt["checkpoint"], "detector_confidence_threshold": 0.15,
                "detector_nms": {"iou_threshold": 0.70, "class_agnostic": False,
                                 "implementation": "Ultralytics predict; explicitly retaining prior defaults"},
                "image_size": 640, "device": rt["device"], "gpu_name": rt["gpu"],
                "detection_count": len(detection_rows), "raw_detection_count": len(detection_rows),
                "estimated_distinct_drone_count": len(distinct_rows),
                "detections": detection_rows[:200], "cleaned_detections": distinct_rows[:200],
                "detections_truncated": len(detection_rows) > 200,
                "tracking": "not applicable to a single image",
                "annotation_render_policy": "retained boxes only; confidence and IDs are in detections.csv to avoid label clutter",
                "formation_prediction": formation["label"],
                "formation_reason": formation["reason"],
                "image_geometry_version": IMAGE_GEOMETRY_VERSION,
                "formation_method": "Conservative image-space geometry with perspective-tolerant ring fitting; no heading or temporal evidence is inferred",
                "formation_positions_used": formation["num_positions"],
                "formation_positions_excluded_for_overlap": formation["positions_excluded_for_overlap"],
                "formation_quality": formation["quality"],
                "formation_candidate_scores": formation["candidate_scores"],
                "duplicate_suppressions": duplicate_suppressions,
                "duplicate_suppression_policy": {
                    "iou_min": IMAGE_DUPLICATE_IOU_MIN,
                    "intersection_over_smaller_min": IMAGE_DUPLICATE_SMALLER_BOX_COVERAGE_MIN,
                    "center_distance_max_fraction_of_smaller_box_diagonal": IMAGE_DUPLICATE_CENTER_DIAGONAL_RATIO_MAX,
                    "representative": "highest confidence; detector box order breaks ties",
                },
                "overlapping_detection_pairs": unresolved_overlap_pairs,
                "artifacts": {"annotated_image": "annotated.png", "detections_csv": "detections.csv", "metadata_json": "metadata.json"}}
    (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "files": {"annotated_image": f"/api/jobs/{job_id}/annotated.png",
                                   "detections_csv": f"/api/jobs/{job_id}/detections.csv",
                                   "metadata_json": f"/api/jobs/{job_id}/metadata.json"}}


def _suppress_duplicate_image_boxes(boxes_xyxy, confidences):
    """Drop only near-identical, heavily overlapping boxes for still-image geometry.

    This is a conservative second pass after YOLO's class-aware NMS. Nearby
    drones remain separate unless one box covers almost all of the other and
    their centers are very close relative to the smaller box size.
    """
    if len(boxes_xyxy) != len(confidences):
        raise ValueError("boxes and confidences must have equal lengths")
    order = sorted(range(len(boxes_xyxy)), key=lambda i: (-float(confidences[i]), i))
    kept = []
    suppressed = []
    for candidate in order:
        x1, y1, x2, y2 = map(float, boxes_xyxy[candidate])
        area_candidate = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate_of = None
        duplicate_metrics = None
        if area_candidate > 0:
            for representative in kept:
                a1, b1, a2, b2 = map(float, boxes_xyxy[representative])
                area_rep = max(0.0, a2 - a1) * max(0.0, b2 - b1)
                if area_rep <= 0:
                    continue
                iw = max(0.0, min(x2, a2) - max(x1, a1))
                ih = max(0.0, min(y2, b2) - max(y1, b1))
                intersection = iw * ih
                if intersection <= 0:
                    continue
                union = area_candidate + area_rep - intersection
                iou = intersection / max(union, 1e-9)
                smaller_coverage = intersection / min(area_candidate, area_rep)
                center_candidate = ((x1 + x2) / 2, (y1 + y2) / 2)
                center_rep = ((a1 + a2) / 2, (b1 + b2) / 2)
                center_distance = ((center_candidate[0] - center_rep[0]) ** 2 +
                                   (center_candidate[1] - center_rep[1]) ** 2) ** 0.5
                smaller_diagonal = min(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5,
                                       ((a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5)
                center_ratio = center_distance / max(smaller_diagonal, 1e-9)
                if (iou >= IMAGE_DUPLICATE_IOU_MIN and
                        smaller_coverage >= IMAGE_DUPLICATE_SMALLER_BOX_COVERAGE_MIN and
                        center_ratio <= IMAGE_DUPLICATE_CENTER_DIAGONAL_RATIO_MAX):
                    duplicate_of = representative
                    duplicate_metrics = {"iou": round(iou, 6),
                                         "intersection_over_smaller": round(smaller_coverage, 6),
                                         "center_distance_px": round(center_distance, 3),
                                         "center_distance_smaller_diagonal_ratio": round(center_ratio, 6)}
                    break
        if duplicate_of is None:
            kept.append(candidate)
        else:
            suppressed.append({"detection_id": candidate + 1,
                               "representative_detection_id": duplicate_of + 1,
                               "confidence": float(confidences[candidate]),
                               "representative_confidence": float(confidences[duplicate_of]),
                               **duplicate_metrics})
    return sorted(kept), sorted(suppressed, key=lambda item: item["detection_id"])


def _assess_image_formation(points, boxes_xyxy, classify, confidences=None):
    """Assess geometry while excluding only the weaker member of severe overlaps.

    Raw detection records and distinct-count estimates are left untouched. A
    heavily overlapping pair is ambiguous, so the lower-confidence member is
    omitted from spatial fitting rather than invalidating every other point.
    """
    overlap_pairs = []
    for i, a in enumerate(boxes_xyxy):
        for j in range(i + 1, len(boxes_xyxy)):
            b = boxes_xyxy[j]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            if inter / max(area_a + area_b - inter, 1e-9) >= 0.5:
                overlap_pairs.append([i + 1, j + 1])

    excluded = set()
    if confidences is not None and len(confidences) == len(points):
        for left, right in overlap_pairs:
            li, ri = left - 1, right - 1
            # Stable tie break: keep the earlier detector record.
            excluded.add(li if confidences[li] < confidences[ri] else ri)
    elif overlap_pairs:
        return {"label": "Unknown",
                "reason": "overlapping detections cannot be resolved without per-box confidence values.",
                "candidate_scores": {}, "quality": 0.0,
                "num_positions": len(set(map(tuple, points))),
                "positions_excluded_for_overlap": 0,
                "overlapping_detection_pairs": overlap_pairs}
    usable_points = [point for index, point in enumerate(points) if index not in excluded]
    if len(set(map(tuple, usable_points))) < 4:
        return {"label": "Unknown", "reason": "too few unambiguous drone positions remain after overlap review.",
                "candidate_scores": {}, "quality": 0.0, "num_positions": len(set(map(tuple, usable_points))),
                "positions_excluded_for_overlap": len(excluded),
                "overlapping_detection_pairs": overlap_pairs}
    geometry = classify(usable_points)
    reason = geometry.rejection_reason or "Image-space geometry matches a spatial template; this is an exploratory candidate, not verified real-world formation accuracy."
    if excluded:
        reason = f"Excluded {len(excluded)} lower-confidence box(es) from geometry because of severe overlap. {reason}"
    return {"label": geometry.label.removesuffix(" formation") if geometry.label != "Unknown" else "Unknown",
            "reason": reason,
            "candidate_scores": geometry.candidate_scores, "quality": geometry.quality,
            "num_positions": geometry.num_positions,
            "positions_excluded_for_overlap": len(excluded),
            "overlapping_detection_pairs": overlap_pairs}


def _process_video_serialized(input_path: Path, job_dir: Path, filename: str, job_id: str):
    with _inference_lock:
        return _process_video(input_path, job_dir, filename, job_id)


def _process_video(input_path: Path, job_dir: Path, filename: str, job_id: str):
    rt = _load_runtime()
    cv2, torch = rt["cv2"], rt["torch"]
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the uploaded video; check that it is a valid supported video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video metadata is invalid (FPS or dimensions unavailable).")
    if fps < 10:
        cap.release()
        raise ValueError("Source video must be at least 10 FPS for the 10 Hz temporal geometry analysis.")
    if declared and declared / fps > 300:
        cap.release()
        raise ValueError("Videos longer than five minutes are not accepted in this prototype.")

    video_path = job_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not create the annotated MP4 output.")
    detector = rt["detector"]
    # The cached detector is shared to avoid reloading weights. Serialize jobs
    # and reset tracker history before each uploaded video.
    predictor = getattr(detector, "predictor", None)
    if predictor is not None:
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
        predictor.vid_path = [None] * len(getattr(predictor, "trackers", []))
    target_sampling_hz = 10.0
    detections_path = job_dir / "detections.csv"
    formations_path = job_dir / "formations.csv"
    windows = []
    frame_index = 0
    sampled_index = 0
    window_frames = []
    window_camera_affines = []
    previous_sample_gray = None
    previous_sample_boxes = []
    window_id = 0
    detection_count = 0
    frames_with_detections = 0
    tracked_detection_count = 0
    track_ids = set()
    detection_preview = []
    sample_source_frames = []
    colors = [(50, 220, 170), (255, 160, 60), (80, 190, 255), (210, 100, 255), (80, 230, 255), (255, 100, 120)]
    det_fields = ["frame_number", "time_seconds", "class_id", "class_name", "track_id", "confidence", "x1", "y1", "x2", "y2"]
    try:
        with detections_path.open("w", newline="", encoding="utf-8") as detections_file, formations_path.open("w", newline="", encoding="utf-8") as formations_file:
            dw = csv.DictWriter(detections_file, fieldnames=det_fields); dw.writeheader()
            formation_fields = ["window", "sample_start", "sample_end", "prediction", "reason", "geometry_quality", "usable_tracks", "geometry_stability_error", "geometry_agreement", "heading_reliable", "heading_coherence", "direction_angle_deg", "displacement_ratio", "camera_motion_reliable"]
            fw = csv.DictWriter(formations_file, fieldnames=formation_fields); fw.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index >= int(fps * 300):
                    raise ValueError("Video exceeds the five-minute processing limit.")
                sample_due = frame_index >= int(round(sampled_index * fps / target_sampling_hz))
                sample_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if sample_due else None
                result = detector.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15,
                                        imgsz=640, verbose=False, device=rt["device"])[0]
                boxes = result.boxes
                ids = boxes.id.int().cpu().tolist() if boxes is not None and boxes.id is not None else []
                frame_records = []
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    for idx, (coords, conf, cls) in enumerate(zip(xyxy, confs, classes)):
                        if int(cls) != 0:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in coords]
                        tid = ids[idx] if idx < len(ids) else None
                        frame_records.append((x1, y1, x2, y2, float(conf), tid))
                        if tid is not None:
                            track_ids.add(int(tid))
                            tracked_detection_count += 1
                        detection_count += 1
                        detection_row = {"frame_number": frame_index, "time_seconds": round(frame_index / fps, 4), "class_id": 0, "class_name": "drone", "track_id": "" if tid is None else tid, "confidence": round(float(conf), 6), "x1": round(x1, 3), "y1": round(y1, 3), "x2": round(x2, 3), "y2": round(y2, 3)}
                        dw.writerow(detection_row)
                        if len(detection_preview) < 30:
                            detection_preview.append(detection_row)
                        color = colors[(int(tid) if tid is not None else idx) % len(colors)]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"drone {float(conf):.2f}" + (f" | ID {tid}" if tid is not None else " | ID pending")
                        cv2.putText(frame, label, (int(x1), max(20, int(y1) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
                if frame_records:
                    frames_with_detections += 1
                # Select the nearest source frame for each 100 ms geometry sample.
                # This avoids a fixed integer stride drifting on 25/29.97 FPS inputs.
                if sample_due:
                    current_boxes = [[r[0], r[1], r[2], r[3]] for r in frame_records]
                    frame_observations = []
                    for x1, y1, x2, y2, conf, tid in frame_records:
                        frame_observations.append({"track_id": None if tid is None else str(tid),
                                                   "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                                                   "confidence": conf})
                    window_frames.append(frame_observations)
                    affine = None if previous_sample_gray is None else _estimate_background_affine(
                        cv2, previous_sample_gray, sample_gray, previous_sample_boxes, current_boxes)
                    window_camera_affines.append(affine)
                    previous_sample_gray = sample_gray
                    previous_sample_boxes = current_boxes
                    sample_source_frames.append(frame_index)
                    sampled_index += 1
                    if sampled_index % 20 == 0:
                        window_id += 1
                        assessment = rt["geometry_classify_video"](window_frames, window_camera_affines, width, height, sample_period=0.1)
                        camera_reliable = bool(window_camera_affines and all(x and x.get("reliable") for x in window_camera_affines[1:]))
                        row = {"window": window_id, "sample_start": sample_source_frames[-20], "sample_end": sample_source_frames[-1],
                               **assessment, "camera_motion_reliable": camera_reliable}
                        fw.writerow(row); windows.append(row)
                        window_frames, window_camera_affines, sample_source_frames = [], [], []
                        previous_sample_gray, previous_sample_boxes = None, []
                writer.write(frame)
                frame_index += 1
    finally:
        cap.release(); writer.release()
    if frame_index == 0:
        raise ValueError("Video contains no readable frames.")
    # Preserve abstention when fewer than a full 20-sample (2 second) window remains.
    remaining_samples = sampled_index % 20
    if remaining_samples:
        row = {"window": window_id + 1, "sample_start": sample_source_frames[0], "sample_end": sample_source_frames[-1],
               "prediction": "Unknown", "reason": f"incomplete observation window ({remaining_samples}/20 sampled frames)",
               "geometry_quality": "", "usable_tracks": "", "geometry_stability_error": "", "geometry_agreement": "",
               "heading_reliable": False, "heading_coherence": "", "direction_angle_deg": "", "displacement_ratio": "", "camera_motion_reliable": False}
        with formations_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=formation_fields).writerow(row)
        windows.append(row)
    metadata = {"job_id": job_id, "input_filename": filename, "source_fps": fps, "width": width, "height": height,
                "frames_processed": frame_index, "duration_seconds": round(frame_index / fps, 3), "target_sampling_fps": target_sampling_hz,
                "actual_sampling_fps": round(sampled_index / max(frame_index / fps, 1e-9), 6),
                "sampling_warning": "Geometry observations are selected at the nearest available source frame to each 100 ms target; source-frame timing is quantized." if fps % 10 else None,
                "detector_checkpoint": rt["checkpoint"], "detector_confidence_threshold": 0.15,
                "tracker": "Ultralytics ByteTrack (bytetrack.yaml)", "image_size": 640,
                "formation_method": "real-image spatial geometry plus camera-compensated temporal geometry; no learned or synthetic-trained formation model",
                "formation_geometry_version": VIDEO_GEOMETRY_VERSION,
                "formation_policy": "spatial classes require persistent geometry; directional classes require reliable background compensation, persistent tracks, and coherent travel heading",
                "prediction_scores": "geometry quality heuristic; not a probability",
                "device": rt["device"], "gpu_name": rt["gpu"],
                "detection_count": detection_count, "frames_with_detections": frames_with_detections,
                "tracked_detection_count": tracked_detection_count,
                "unique_track_ids_observed": len(track_ids), "complete_windows": window_id, "windows": windows,
                "detection_preview": detection_preview,
                "artifacts": {"video": "annotated.mp4", "detections_csv": "detections.csv", "formations_csv": "formations.csv"}}
    (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "files": {"video": f"/api/jobs/{job_id}/annotated.mp4",
                                   "detections_csv": f"/api/jobs/{job_id}/detections.csv",
                                   "formations_csv": f"/api/jobs/{job_id}/formations.csv",
                                   "metadata_json": f"/api/jobs/{job_id}/metadata.json"}}


def _estimate_background_affine(cv2, previous_gray, current_gray, previous_boxes, current_boxes):
    """Estimate prev->current camera transform from background optical flow.

    Drone boxes are excluded in both frames. The result is explicitly marked
    unreliable unless there are enough spatially distributed RANSAC inliers.
    """
    height, width = previous_gray.shape[:2]
    mask = __import__("numpy").full((height, width), 255, dtype="uint8")
    for boxes in (previous_boxes, current_boxes):
        for x1, y1, x2, y2 in boxes:
            pad_x = max(4, int((x2 - x1) * 0.8))
            pad_y = max(4, int((y2 - y1) * 0.8))
            left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
            right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
            cv2.rectangle(mask, (left, top), (right, bottom), 0, -1)
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=500, qualityLevel=0.01,
                                     minDistance=8, mask=mask, blockSize=7)
    if points is None or len(points) < 20:
        return {"reliable": False, "matrix": None, "matches": 0, "inliers": 0, "reason": "too few background features"}
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None,
                                                  winSize=(21, 21), maxLevel=3)
    if moved is None or status is None:
        return {"reliable": False, "matrix": None, "matches": 0, "inliers": 0, "reason": "optical flow failed"}
    good_prev = points[status.reshape(-1) == 1].reshape(-1, 2)
    good_curr = moved[status.reshape(-1) == 1].reshape(-1, 2)
    # Exclude endpoints landing inside a drone box in the current frame.
    keep = __import__("numpy").ones(len(good_curr), dtype=bool)
    for x1, y1, x2, y2 in current_boxes:
        pad_x, pad_y = max(4, (x2 - x1) * 0.8), max(4, (y2 - y1) * 0.8)
        keep &= ~((good_curr[:, 0] >= x1 - pad_x) & (good_curr[:, 0] <= x2 + pad_x) &
                  (good_curr[:, 1] >= y1 - pad_y) & (good_curr[:, 1] <= y2 + pad_y))
    good_prev, good_curr = good_prev[keep], good_curr[keep]
    if len(good_prev) < 20:
        return {"reliable": False, "matrix": None, "matches": int(len(good_prev)), "inliers": 0, "reason": "too few background flow matches"}
    matrix, inlier_mask = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC,
                                                      ransacReprojThreshold=2.5, maxIters=2000,
                                                      confidence=0.99, refineIters=10)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    ratio = inliers / max(len(good_prev), 1)
    reliable = matrix is not None and inliers >= 15 and ratio >= 0.50
    return {"reliable": bool(reliable),
            "matrix": matrix.tolist() if reliable else None,
            "matches": int(len(good_prev)), "inliers": inliers,
            "inlier_ratio": round(ratio, 6),
            "reason": "" if reliable else "background affine lacked enough RANSAC support"}


@app.get("/api/jobs/{job_id}/{filename}")
def get_job_file(job_id: str, filename: str):
    try:
        uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(404, "Job not found") from exc
    if filename not in {"annotated.mp4", "annotated.png", "detections.csv", "formations.csv", "metadata.json"}:
        raise HTTPException(404, "File not found")
    # Video and still-image jobs are stored separately. Keep the UUID and
    # filename allowlists above, then search only those two fixed job roots.
    for job_root in ("real_video_jobs", "real_image_jobs"):
        path = ROOT / "outputs" / job_root / job_id / filename
        if path.is_file():
            return FileResponse(path)
    raise HTTPException(404, "File not found")
