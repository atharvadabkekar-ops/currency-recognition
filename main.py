from src.dataset import get_dataloaders
from src.train import train_model
from src.utils import print_gpu_info
import argparse
from src.config import EPOCHS

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test"], required=True)
parser.add_argument("--epochs", type=int, default=EPOCHS)
args = parser.parse_args()

def main():
    print_gpu_info()
    train_loader, val_loader = get_dataloaders()
    train_model(train_loader, val_loader)

if __name__ == "__main__":
    main()
