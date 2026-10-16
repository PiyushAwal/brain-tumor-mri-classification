"""Train the custom CNN and the transfer-learning models.

Examples
    python train.py                              # all five models
    python train.py --models custom_cnn resnet50 # a subset
    python train.py --epochs 3 --fine-tune-epochs 1   # quick smoke test
"""
import argparse
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import config
import data_utils
import evaluate as ev
import model_zoo as zoo

KEYS = ("accuracy", "val_accuracy", "loss", "val_loss")


def callbacks(ckpt_path, threshold=None):
    return [
        EarlyStopping("val_loss", patience=config.PATIENCE, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True,
                        initial_value_threshold=threshold),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def train_one(name, train_ds, val_ds, test_ds, class_names, class_weights, epochs, ft_epochs):
    tf.keras.backend.clear_session()
    ckpt = str(config.CKPT_DIR / f"{name}.keras")  # Keras 3 checkpoints need .keras
    model = zoo.build_model(name, len(class_names))
    compile_kw = dict(loss="categorical_crossentropy", metrics=["accuracy"])

    # Phase 1: custom CNN trains fully; transfer models train only the new head
    model.compile(optimizer=Adam(1e-3), **compile_kw)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                   class_weight=class_weights, callbacks=callbacks(ckpt), verbose=2)
    history = {k: [float(v) for v in h1.history[k]] for k in KEYS}
    switch = None

    # Phase 2: fine-tune the top backbone layers with a small learning rate
    if name != "custom_cnn" and ft_epochs > 0:
        best = float(min(h1.history["val_loss"]))
        zoo.unfreeze_top(zoo.get_backbone(model))
        model.compile(optimizer=Adam(1e-5), **compile_kw)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                       class_weight=class_weights, callbacks=callbacks(ckpt, best), verbose=2)
        switch = len(history["loss"])
        for k in KEYS:
            history[k] += [float(v) for v in h2.history[k]]

    (config.REPORTS_DIR / f"{name}_history.json").write_text(
        json.dumps({"history": history, "finetune_start_epoch": switch}, indent=2))

    # Export the best checkpoint to .h5 (project deliverable), reload it and evaluate that file
    h5_path = config.MODELS_DIR / f"{name}.h5"
    tf.keras.models.load_model(ckpt, compile=False).save(h5_path)
    saved = tf.keras.models.load_model(h5_path, compile=False)
    metrics = ev.evaluate_model(saved, test_ds, class_names, h5_path, name)
    ev.save_metrics(metrics)
    print(f"[{name}] test acc={metrics['accuracy']:.4f}  macro-F1={metrics['f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODEL_NAMES, choices=config.MODEL_NAMES)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    config.BATCH_SIZE = args.batch_size
    tf.keras.utils.set_random_seed(config.SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    train_ds, val_ds, test_ds, class_names, weights = data_utils.load_datasets()
    (config.REPORTS_DIR / "class_names.json").write_text(json.dumps(class_names))
    print("Classes:", class_names, "| class weights:", {k: round(v, 2) for k, v in weights.items()})

    for name in args.models:
        print(f"\n=== Training {name} ===")
        try:
            train_one(name, train_ds, val_ds, test_ds, class_names, weights,
                      args.epochs, args.fine_tune_epochs)
        except Exception as exc:  # keep going so one failure doesn't waste the whole run
            print(f"[{name}] FAILED: {exc}")

    print(ev.build_comparison())


if __name__ == "__main__":
    main()
