import numpy as np

data = np.load("octmnist_64.npz")

for split in ["train", "val", "test"]:
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"]

    print(f"{split}: {images.shape}, labels: {labels.shape}")