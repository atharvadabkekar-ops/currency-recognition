import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_DIR,
)
from src.model import build_model


def train_model(train_loader, val_loader):
    print("[INFO] Starting training...")

    # ===== Model =====
    model = build_model().to(DEVICE)

    # ===== Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # ===== AMP Scaler =====
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        # ================= TRAIN =================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total

        # ================= VALIDATION =================
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # ================= SAVE BEST =================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_model.pth"
            )
            print("[INFO] Best model saved.")

    print("\n[INFO] Training complete.")
