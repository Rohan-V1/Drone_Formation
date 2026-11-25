"""Build a short local test clip from original UAVSwarm train images; never overwrites."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "UAVSwarm-dataset-master" / "UAVSwarm-dataset-master" / "train" / "UAVSwarm-29" / "img1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="New MP4 path (must not exist)")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite test clip: {output}")
    if not 20 <= args.frames <= 120:
        raise ValueError("Test clip must contain between 20 and 120 frames")
    images = sorted(SOURCE.glob("*.jpg"))[: args.frames]
    if len(images) < 20:
        raise FileNotFoundError(f"Expected at least 20 original train frames at {SOURCE}")
    output.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Cannot decode source image {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write test MP4 at {output}")
    count = 0
    try:
        for path in images:
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Unreadable or dimension-mismatched image: {path}")
            writer.write(image)
            count += 1
    finally:
        writer.release()
    check = cv2.VideoCapture(str(output))
    decoded = 0
    try:
        while True:
            ok, _ = check.read()
            if not ok:
                break
            decoded += 1
    finally:
        check.release()
    if decoded != count:
        raise RuntimeError(f"Test clip decoded {decoded}/{count} frames")
    print(f"Created and decoded {count} frames at {width}x{height}, 30 FPS: {output}")


if __name__ == "__main__":
    main()
