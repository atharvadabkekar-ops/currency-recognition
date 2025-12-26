from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.config import TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.6, 1.0)
    ),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),
    transforms.RandomPerspective(
        distortion_scale=0.3,
        p=0.3
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 2.0)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders():
    train_dataset = ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = ImageFolder(
        root=VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader

