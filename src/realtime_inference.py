import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model import build_model
from src.config import DEVICE, IMAGE_SIZE, CLASSES, CHECKPOINT_DIR

# ===== Load model =====
model = build_model().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== Phone camera stream =====
CAMERA_URL = "http://100.114.194.163:8080/video"  
cap = cv2.VideoCapture(CAMERA_URL)

assert cap.isOpened(), "❌ Cannot open camera stream"

print("📷 Real-time inference started (press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

        conf = conf.item()
        pred = pred.item()

    if conf < 0.40:
        label = "Unknown"
    else:
        label = f"Rs.{CLASSES[pred]}"

    print(probs)
    # Display
    cv2.putText(
        frame,
        f"Prediction: Rs{label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Indian Currency Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()
