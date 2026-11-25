"""Evaluate the current detector on the real reviewed Seraphim-build val set."""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(r"C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt")
DATA = PROJECT / "datasets/external_seraphim_detector_v1/prepared/data.yaml"
OUTPUT = PROJECT / "outputs/seraphim_detector_build/run_004/current_checkpoint_val"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {OUTPUT}")
    if not WEIGHTS.is_file() or not DATA.is_file():
        raise FileNotFoundError(f"Missing weights or data YAML: {WEIGHTS}, {DATA}")
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA), imgsz=640, batch=8, device=0, split="val",
        project=str(OUTPUT.parent), name=OUTPUT.name, exist_ok=False, plots=False,
    )
    report = {
        "weights": str(WEIGHTS),
        "dataset_yaml": str(DATA),
        "evaluated_split": "manually reviewed real 30-image pilot validation split",
        "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "class_names": model.names,
        "output_dir": str(OUTPUT),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
