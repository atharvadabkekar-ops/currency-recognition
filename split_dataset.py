import os
import shutil
import random
from pathlib import Path

# ================= CONFIG =================
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

SPLIT_RATIO = 0.8   # 80% train, 20% val
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# ==========================================

random.seed(SEED)

def split_class(class_name):
    raw_class_dir = RAW_DIR / class_name
    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    images = [
        f for f in raw_class_dir.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ]

    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        shutil.copy(img, train_class_dir / img.name)

    for img in val_images:
        shutil.copy(img, val_class_dir / img.name)

    print(f"{class_name}: {len(train_images)} train | {len(val_images)} val")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError("data/raw folder not found")

    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)

    classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]

    print("Splitting dataset...\n")
    for cls in classes:
        split_class(cls)

    print("\n✅ Dataset split complete!")


if __name__ == "__main__":
    main()
