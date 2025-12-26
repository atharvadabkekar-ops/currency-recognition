"""
config.py
Global configuration file for Indian currency classification project.
"""

import torch
from pathlib import Path

# ==========================
# ⚙ DEVICE SELECTOR
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ==========================
# 📁 PROJECT PATHS
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
TRAIN_DIR       = DATA_DIR / "train"
VAL_DIR         = DATA_DIR / "val"

CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR      = PROJECT_ROOT / "outputs"

# Create directories if missing
for folder in [CHECKPOINT_DIR, OUTPUT_DIR]:
    folder.mkdir(exist_ok=True)

# ==========================
# 🖼 IMAGE SETTINGS
# ==========================
IMAGE_SIZE      = 256
BATCH_SIZE      = 64

# ==========================
# 📚 TRAINING SETTINGS
# ==========================
EPOCHS          = 10
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
NUM_WORKERS     = 8     # for dataloaders, adjust based on CPU cores

# ==========================
# 💰 CURRENCY CLASSES
# ==========================
CLASSES = ["10", "20", "50", "100", "200", "500", "2000"]
NUM_CLASSES = len(CLASSES)

# ==========================
# 🧪 SAFETY CHECKS
# ==========================
assert TRAIN_DIR.exists(), "[ERROR] Missing training folder!"
assert VAL_DIR.exists(),   "[ERROR] Missing validation folder!"

print("[CONFIG] Loaded successfully.")
