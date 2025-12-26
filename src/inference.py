import torch
from PIL import Image
from torchvision import transforms
from src.model import build_model
from src.config import DEVICE, IMAGE_SIZE, CLASSES, CHECKPOINT_DIR

def predict_image(image_path):
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth"))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()

    return CLASSES[pred]
