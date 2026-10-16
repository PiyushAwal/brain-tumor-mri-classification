"""Dataset discovery, tf.data pipelines, augmentation and class weights.

TensorFlow is imported lazily so the Streamlit dashboard can use the file-system
helpers (list_images, class_counts, get_split_dirs) without loading TF.
"""
from pathlib import Path

import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------- file-system helpers
def _child_dirs(folder):
    try:
        return [p for p in Path(folder).iterdir() if p.is_dir()]
    except OSError:
        return []


def _find_split(names):
    """Find a split folder (train/valid/test) directly in data/ or one folder below it.

    Each split is searched independently, so Google-Drive style layouts such as
    data/train-2026...-001/train/<class>/ and data/test-2026...-001/test/<class>/ work.
    """
    root = config.DATA_DIR
    if not root.exists():
        return None
    level1 = _child_dirs(root)
    for p in level1:
        if p.name.lower() in names:
            return p
    for base in level1:
        for p in _child_dirs(base):
            if p.name.lower() in names:
                return p
    return None


def get_split_dirs():
    """Return {'train': Path|None, 'val': Path|None, 'test': Path|None}."""
    return {
        "train": _find_split(config.TRAIN_DIR_NAMES),
        "val": _find_split(config.VAL_DIR_NAMES),
        "test": _find_split(config.TEST_DIR_NAMES),
    }


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def class_counts(split_dir):
    return {d.name: len(list_images(d)) for d in sorted(Path(split_dir).iterdir()) if d.is_dir()}


def compute_class_weights(counts, class_names):
    """Inverse-frequency weights to counter class imbalance."""
    total, n = sum(counts[c] for c in class_names), len(class_names)
    return {i: total / (n * max(counts[c], 1)) for i, c in enumerate(class_names)}


# ----------------------------------------------------------------- tf.data pipelines
def _make_ds(directory, shuffle, subset=None, class_names=None):
    import tensorflow as tf

    kwargs = dict(
        labels="inferred", label_mode="categorical", image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, seed=config.SEED, shuffle=shuffle,
    )
    if subset:
        kwargs.update(validation_split=config.VAL_SPLIT, subset=subset)
    if class_names:
        kwargs["class_names"] = class_names
    return tf.keras.utils.image_dataset_from_directory(directory, **kwargs)


def build_augmenter():
    """Rotation, flips, zoom, shifts and brightness (inputs already scaled to 0-1)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08, fill_mode="nearest"),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.2, value_range=(0, 1)),
        ],
        name="augmentation",
    )


def load_datasets(augment=True):
    """Return train_ds, val_ds, test_ds, class_names, class_weights.

    Pixels are normalised to 0-1. Only the training set is augmented.
    """
    import tensorflow as tf

    dirs = get_split_dirs()
    if dirs["train"] is None:
        raise FileNotFoundError(
            f"No train folder found in {config.DATA_DIR}. Expected data/train/<class>/*.jpg"
        )

    if dirs["val"] is not None:
        train_raw = _make_ds(dirs["train"], shuffle=True)
        val_raw = _make_ds(dirs["val"], shuffle=False, class_names=train_raw.class_names)
    else:
        train_raw = _make_ds(dirs["train"], shuffle=True, subset="training")
        val_raw = _make_ds(dirs["train"], shuffle=True, subset="validation")
    class_names = train_raw.class_names
    test_raw = (
        _make_ds(dirs["test"], shuffle=False, class_names=class_names)
        if dirs["test"] is not None else val_raw
    )

    autotune = tf.data.AUTOTUNE
    norm = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)  # noqa: E731
    train_ds = train_raw.map(norm, autotune)
    if augment:
        aug = build_augmenter()
        train_ds = train_ds.map(
            lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 1.0), y), autotune
        )
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_raw.map(norm, autotune).prefetch(autotune)
    test_ds = test_raw.map(norm, autotune).prefetch(autotune)

    weights = compute_class_weights(class_counts(dirs["train"]), class_names)
    return train_ds, val_ds, test_ds, class_names, weights


if __name__ == "__main__":  # quick dataset sanity check: python data_utils.py
    print("Looking in:", config.DATA_DIR, "(exists)" if config.DATA_DIR.exists() else "(DOES NOT EXIST)")
    if config.DATA_DIR.exists():
        print("Contents :", [p.name for p in config.DATA_DIR.iterdir()][:10])
    for split, path in get_split_dirs().items():
        print(f"{split:5s} ->", path if path else "not found")
        if path:
            counts = class_counts(path)
            print("       ", counts, "| total:", sum(counts.values()))