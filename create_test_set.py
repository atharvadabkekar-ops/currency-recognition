import random
import shutil
from pathlib import Path

# ===== CONFIG =====
SOURCE_DIR = Path("data/raw")     # or any folder with images
TEST_DIR = Path("data/test")
IMAGES_PER_CLASS = None           # None = take ALL, or set e.g. 50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==================

TEST_DIR.mkdir(parents=True, exist_ok=True)

for class_dir in SOURCE_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    images = [img for img in class_dir.iterdir()
              if img.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        continue

    random.shuffle(images)

    if IMAGES_PER_CLASS:
        images = images[:IMAGES_PER_CLASS]

    target_class_dir = TEST_DIR / class_dir.name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        shutil.copy(img, target_class_dir / img.name)

    print(f" {class_dir.name}: {len(images)} images copied")

print("\n Test dataset prepared successfully")
