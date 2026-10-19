from medmnist import OCTMNIST
import os

# Create dataset directory
os.makedirs("data", exist_ok=True)

# Download OCTMNIST at 64x64 resolution
dataset = OCTMNIST(
    split="train",
    size=64,
    download=True,
    root="./data"
)

print("OCTMNIST 64x64 downloaded successfully!")