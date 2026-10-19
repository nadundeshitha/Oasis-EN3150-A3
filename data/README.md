# OCTMNIST Dataset

This project uses the **OCTMNIST dataset (64×64 resolution)** from MedMNIST for CNN-based retinal image classification.

## Dataset License and Attribution

This project uses **OCTMNIST (64×64)** from the MedMNIST collection for retinal image classification.

**Dataset source:** https://medmnist.com/

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**License details:** https://creativecommons.org/licenses/by/4.0/

**Reference:**

Yang, J., Shi, R., Wei, D., et al. (2023). *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data, 10, 41.

The dataset is used for educational and research purposes. Original dataset files are not stored in this repository and can be downloaded using the provided Python script.

Any further preprocessing or modifications performed in this project will be documented separately.

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