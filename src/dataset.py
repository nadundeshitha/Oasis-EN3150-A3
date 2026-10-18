from pathlib import Path
import numpy as np
import tensorflow as tf
from medmnist import INFO

SEED = 42
DEFAULT_BATCH_SIZE = 32
NUM_CLASSES = 4


def make_dataset(images, labels, batch_size=DEFAULT_BATCH_SIZE, training=False, seed=SEED):
    """
    Creates a tf.data.Dataset using the exact preprocessing pipeline from
    01_data_preprocessing.ipynb:
      - Reads (64, 64) uint8 images and int32 labels via generator
      - Shuffles with buffer_size=2048 if training=True
      - Batches by batch_size
      - Normalizes pixels to [0, 1] float32 and expands channel dim to (64, 64, 1)
      - Prefetches 1 batch
    """
    def generator():
        for image, label in zip(images, labels):
            yield image, np.int32(label)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if training:
        ds = ds.shuffle(
            buffer_size=2048,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=-1)
        return image, label

    ds = ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.prefetch(1)


def load_octmnist_datasets(batch_size=DEFAULT_BATCH_SIZE, data_dir=None, seed=SEED):
    """
    Loads data/octmnist_64.npz and data/split_indices_seed42.npz, reconstructs
    the exact partitions from 01_data_preprocessing.ipynb:
      - X_train, y_train (76,128 samples)
      - X_val_eval, y_val_eval (15,170 deduplicated validation samples)
      - X_test_eval, y_test_eval (15,172 deduplicated test samples)
    and returns:
      (train_ds, val_ds, test_ds), (X_train, y_train, X_val_eval, y_val_eval, X_test_eval, y_test_eval), info
    """
    tf.random.set_seed(seed)

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    dataset_path = data_dir / "octmnist_64.npz"
    split_path = data_dir / "split_indices_seed42.npz"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at {dataset_path}. "
            "Run `python data/download_octmnist.py` first."
        )
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split indices archive not found at {split_path}. "
            "Run `notebooks/01_data_preprocessing.ipynb` first."
        )

    data = np.load(dataset_path)
    saved = np.load(split_path)

    X_all = np.concatenate(
        [
            data["train_images"],
            data["val_images"],
            data["test_images"],
        ],
        axis=0,
    )

    y_all = np.concatenate(
        [
            data["train_labels"],
            data["val_labels"],
            data["test_labels"],
        ],
        axis=0,
    ).flatten()

    train_idx = saved["train_idx"]
    val_eval_idx = saved["val_eval_idx"]
    test_eval_idx = saved["test_eval_idx"]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val_eval, y_val_eval = X_all[val_eval_idx], y_all[val_eval_idx]
    X_test_eval, y_test_eval = X_all[test_eval_idx], y_all[test_eval_idx]

    train_ds = make_dataset(X_train, y_train, batch_size=batch_size, training=True, seed=seed)
    val_ds = make_dataset(X_val_eval, y_val_eval, batch_size=batch_size, training=False, seed=seed)
    test_ds = make_dataset(X_test_eval, y_test_eval, batch_size=batch_size, training=False, seed=seed)

    info = INFO["octmnist"]

    return (
        (train_ds, val_ds, test_ds),
        (X_train, y_train, X_val_eval, y_val_eval, X_test_eval, y_test_eval),
        info,
    )
