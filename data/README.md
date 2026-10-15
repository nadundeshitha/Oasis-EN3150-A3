# OCTMNIST Dataset

This project uses the **OCTMNIST dataset (64×64 resolution)** from MedMNIST for CNN-based retinal image classification.

## Download Instructions

1. Install the required package:

   ```bash
   python -m pip install medmnist
   ```

2. Run the dataset download script from the data:

   ```bash
   python download_octmnist.py
   ```

3. The dataset will be saved in:

   `data/octmnist_64.npz`

The dataset contains predefined training, validation, and testing splits.

**Note:** Dataset files (`.npz`) are excluded from Git tracking. Each team member should download the dataset locally using the provided script.