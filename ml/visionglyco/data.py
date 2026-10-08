"""Label loading, patient-level splitting and tf.data pipelines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import CROP_FRACTION, IMAGE_SIZE, TASK_CLASSIFY
from .preprocess import load_eye_image

REQUIRED_COLUMNS = ("image", "patient_id")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Split:
    paths: np.ndarray
    targets: np.ndarray
    labels: np.ndarray  # binary labels for classification, raw targets for regression
    patients: np.ndarray

    def __len__(self):
        return len(self.paths)


def load_labels(csv_path, images_dir, target_column):
    """Read the labels CSV and resolve each image path.

    The CSV needs one row per image with columns `image` (path relative to
    images_dir), `patient_id`, and the measured target (e.g. `hba1c` in %, or
    `glucose_mg_dl`) taken at the same visit as the photo.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in (*REQUIRED_COLUMNS, target_column) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    df = df.dropna(subset=[*REQUIRED_COLUMNS, target_column]).copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df[target_column] = pd.to_numeric(df[target_column], errors="raise")

    images_dir = Path(images_dir)
    df["path"] = [str(images_dir / p) for p in df["image"]]
    bad_ext = df[~df["path"].str.lower().str.endswith(tuple(IMAGE_EXTENSIONS))]
    if len(bad_ext):
        raise ValueError(f"Unsupported image types, e.g. {bad_ext['path'].iloc[0]}")
    missing_files = [p for p in df["path"] if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} images listed in {csv_path} were not found, "
            f"e.g. {missing_files[0]}"
        )
    if df.empty:
        raise ValueError(f"No labelled rows in {csv_path}")
    return df.reset_index(drop=True)


def make_labels(targets, task, threshold):
    if task == TASK_CLASSIFY:
        return (targets >= threshold).astype(np.int32)
    return targets.astype(np.float32)


def split_by_patient(df, target_column, task, threshold, val_fraction, test_fraction, seed):
    """Split so that no patient appears in more than one of train/val/test.

    Several photos of the same eye are near-duplicates; letting them straddle
    the split inflates test accuracy without the model learning anything real.
    """
    targets = df[target_column].to_numpy(dtype=np.float32)
    labels = make_labels(targets, task, threshold)
    groups = df["patient_id"].to_numpy()
    paths = df["path"].to_numpy()
    indices = np.arange(len(df))

    def hold_out(idx, fraction, rng_seed):
        if task == TASK_CLASSIFY:
            n_splits = max(2, int(round(1 / fraction)))
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
            keep, held = next(splitter.split(idx, labels[idx], groups[idx]))
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=rng_seed)
            keep, held = next(splitter.split(idx, groups=groups[idx]))
        return idx[keep], idx[held]

    train_val, test = hold_out(indices, test_fraction, seed)
    train, val = hold_out(train_val, val_fraction / (1 - test_fraction), seed + 1)

    def build(idx):
        return Split(paths[idx], targets[idx], labels[idx], groups[idx])

    splits = {"train": build(train), "val": build(val), "test": build(test)}
    _assert_disjoint_patients(splits)
    return splits


def _assert_disjoint_patients(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a].patients) & set(splits[b].patients)
            if overlap:
                raise AssertionError(f"Patients shared between {a} and {b}: {sorted(overlap)[:5]}")


def _augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=20.0)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0)


def make_dataset(split, batch_size, image_size=IMAGE_SIZE, crop_fraction=CROP_FRACTION,
                 training=False, seed=0):
    def load(path):
        return load_eye_image(path.decode("utf-8"), image_size, crop_fraction).astype(np.float32)

    def load_tf(path, label):
        image = tf.numpy_function(load, [path], tf.float32)
        image.set_shape((image_size, image_size, 3))
        return image, label

    label_dtype = tf.int32 if split.labels.dtype.kind == "i" else tf.float32
    ds = tf.data.Dataset.from_tensor_slices(
        (split.paths.astype(str), tf.constant(split.labels, dtype=label_dtype))
    )
    if training:
        ds = ds.shuffle(len(split), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
