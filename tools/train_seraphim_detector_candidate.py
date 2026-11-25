"""Fine-tune a separate detector on Seraphim train + reviewed real pilot."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(r"C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt")
DATA = PROJECT / "datasets/external_seraphim_detector_v1/prepared/data.yaml"
PREPARATION = PROJECT / "datasets/external_seraphim_detector_v1/prepared/preparation_report.json"
RUN_PROJECT = Path(r"C:\DroneTrainingRuns")
RUN_NAME = "seraphim_detector_finetune03"
REPORT_DIR = PROJECT / "outputs/seraphim_detector_build/run_007"

CONFIG = {
    "imgsz": 640,
    "batch": 4,
    "epochs": 100,
    "patience": 15,
    "seed": 42,
    "deterministic": True,
    "device": 0,
    "workers": 0,
    "cache": False,
    "project": str(RUN_PROJECT),
    "name": RUN_NAME,
    "exist_ok": False,
}


def main() -> None:
    output_run = RUN_PROJECT / RUN_NAME
    manifest_path = REPORT_DIR / "run_manifest.json"
    if output_run.exists():
        raise FileExistsError(f"Refusing to overwrite training output: {output_run}")
    if REPORT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite run report directory: {REPORT_DIR}")
    if not all(p.is_file() for p in (WEIGHTS, DATA, PREPARATION)):
        raise FileNotFoundError(f"Missing checkpoint, data YAML or preparation report: {WEIGHTS}, {DATA}, {PREPARATION}")
    prep = json.loads(PREPARATION.read_text(encoding="utf-8"))
    if prep.get("errors") or prep.get("seraphim_images") != 75134 or prep.get("seraphim_labels") != 75134:
        raise RuntimeError("Prepared Seraphim dataset failed its structural audit")
    if prep.get("official_test_split_used_for_training") or prep.get("test_split_used_for_training_or_validation"):
        raise RuntimeError("Refusing to train with Seraphim test split")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to change approved device")
    free, total = torch.cuda.mem_get_info(0)
    if free < 6 * 1024**3:
        raise RuntimeError(f"CUDA free memory below 6 GiB: {free / 1024**3:.2f} GiB")
    model = YOLO(str(WEIGHTS))
    if model.names != {0: "drone"}:
        raise RuntimeError(f"Checkpoint class names do not match dataset: {model.names}")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "initial_checkpoint": str(WEIGHTS),
        "dataset_yaml": str(DATA),
        "seraphim_split": "train only; official test split excluded",
        "combined_train_images": prep["combined_train_images"],
        "validation_images": prep["real_validation_images"],
        "validation_source": prep["validation_data_source"],
        "config": CONFIG,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_free_at_start_bytes": free,
        "gpu_memory_total_bytes": total,
        "run_directory": str(output_run),
        "source_archive_sha256": prep["archive_sha256"],
        "baseline_metrics": str(PROJECT / "outputs/seraphim_detector_build/run_004/current_checkpoint_val/metrics.json"),
        "training_results": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        result = model.train(data=str(DATA), **CONFIG)
        manifest["status"] = "completed"
        manifest["training_results"] = getattr(result, "save_dir", None).__str__()
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
