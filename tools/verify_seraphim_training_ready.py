"""Preflight the staged Seraphim/real-pilot YOLO dataset and model runtime."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "datasets/external_seraphim_detector_v1/prepared/data.yaml"
WEIGHTS = Path(r"C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt")
RUN = Path(r"C:\DroneTrainingRuns\seraphim_detector_finetune03")


def main() -> None:
    if not DATA.is_file() or not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing data YAML or checkpoint: data={DATA.is_file()}, weights={WEIGHTS.is_file()}")
    train_list = DATA.parent / "train_images.txt"
    val_list = DATA.parent / "val_images.txt"
    train_paths = [Path(line.strip()) for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_paths = [Path(line.strip()) for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(train_paths) != 75224 or len(val_paths) != 30:
        raise ValueError(f"Unexpected image-list counts: train={len(train_paths)}, val={len(val_paths)}")
    check_paths = train_paths[:4] + train_paths[-90:] + val_paths
    missing = [str(p) for p in check_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image paths from manifest: {missing[:10]}")
    expected_label_paths = []
    for image_path in train_paths[-90:] + val_paths:
        text_path = str(image_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        expected_label_paths.append(Path(text_path).with_suffix(".txt"))
    missing_labels = [str(p) for p in expected_label_paths if not p.is_file()]
    if missing_labels:
        raise FileNotFoundError(f"Missing pilot labels: {missing_labels[:10]}")
    resolved = check_det_dataset(str(DATA))
    model = YOLO(str(WEIGHTS))
    props = {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_free_total_bytes": torch.cuda.mem_get_info(0) if torch.cuda.is_available() else None,
        "checkpoint_names": model.names,
        "resolved_train": resolved["train"],
        "resolved_val": resolved["val"],
        "train_image_list_count": len(train_paths),
        "validation_image_list_count": len(val_paths),
        "run_path_exists": RUN.exists(),
        "report_path": PROJECT / "outputs/seraphim_detector_build/run_007",
        "report_path_exists": (PROJECT / "outputs/seraphim_detector_build/run_007").exists(),
    }
    print(json.dumps(props, indent=2, default=str))
    if not torch.cuda.is_available() or model.names != {0: "drone"}:
        raise RuntimeError("CUDA or single-class checkpoint verification failed")
    if RUN.exists():
        raise FileExistsError(f"Refusing to use existing run directory: {RUN}")


if __name__ == "__main__":
    main()
