import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model
