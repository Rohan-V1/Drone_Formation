
from pathlib import Path
import random
import shutil

SOURCE = Path("datasets/detection/Drone_Swarm_Dataset")
DEST = Path("datasets/detection/drone_swarm_yolo")
SEED = 42
TRAIN_RATIO = 0.80

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    images_dir = SOURCE / "images"
    labels_dir = SOURCE / "labels"

    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            raise RuntimeError(f"Missing label for image: {image.name}")

    if DEST.exists() and any(DEST.iterdir()):
        raise RuntimeError(
            f"Destination {DEST} is not empty. "
            "Inspect it before rerunning; refusing to overwrite files."
        )

    random.Random(SEED).shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for split, split_images in (
        ("train", train_images),
        ("val", val_images),
    ):
        output_images = DEST / "images" / split
        output_labels = DEST / "labels" / split

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            label = labels_dir / f"{image.stem}.txt"

            shutil.copy2(image, output_images / image.name)
            shutil.copy2(label, output_labels / label.name)

    print(f"Dataset prepared at: {DEST}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Original dataset unchanged.")


if __name__ == "__main__":
    main()
