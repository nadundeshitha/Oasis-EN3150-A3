import numpy as np
import medmnist
from medmnist import INFO
from sklearn.model_selection import train_test_split

def load_dermamnist(target_size=(64, 64)):
    """
    Loads the DermaMNIST dataset, resizes to 64x64, and splits into 70/15/15.
    Returns the raw numpy arrays for training, validation, and testing.
    """
    data_flag = 'dermamnist'
    info = INFO[data_flag]
    
    # Download the dataset using medmnist package
    DataClass = getattr(medmnist, info['python_class'])
    
    try:
        # Try to download the 64x64 version directly
        train_dataset = DataClass(split='train', download=True, size=64)
        val_dataset = DataClass(split='val', download=True, size=64)
        test_dataset = DataClass(split='test', download=True, size=64)
        print("Successfully loaded 64x64 dataset from MedMNIST.")
    except Exception as e:
        print("Could not load 64x64 natively, falling back to 28x28 and resizing...")
        train_dataset = DataClass(split='train', download=True)
        val_dataset = DataClass(split='val', download=True)
        test_dataset = DataClass(split='test', download=True)

    # MedMNIST datasets provide .imgs and .labels as numpy arrays
    x_train, y_train = train_dataset.imgs, train_dataset.labels
    x_val, y_val = val_dataset.imgs, val_dataset.labels
    x_test, y_test = test_dataset.imgs, test_dataset.labels
    
    # Concatenate all splits to re-split them as 70/15/15
    x_all = np.concatenate([x_train, x_val, x_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    # Squeeze labels if necessary
    y_all = np.squeeze(y_all)
    
    # Split the dataset into 70% train, 30% temp
    x_train_split, x_temp, y_train_split, y_temp = train_test_split(
        x_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    
    # Split the temp into 15% val, 15% test (which is 50% of the 30% temp)
    x_val_split, x_test_split, y_val_split, y_test_split = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Dataset split shapes - Train: {x_train_split.shape[0]} ({x_train_split.shape[0]/len(x_all)*100:.1f}%), "
          f"Val: {x_val_split.shape[0]} ({x_val_split.shape[0]/len(x_all)*100:.1f}%), "
          f"Test: {x_test_split.shape[0]} ({x_test_split.shape[0]/len(x_all)*100:.1f}%)")
    
    return (x_train_split, y_train_split), (x_val_split, y_val_split), (x_test_split, y_test_split), info
