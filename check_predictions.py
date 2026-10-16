"""Verify a trained model on the test set: per-image results, per-class accuracy, confusion matrix.

    python check_predictions.py                      # best saved model
    python check_predictions.py --model mobilenetv2 --show 20
"""
import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import config
import data_utils


def pick_default_model():
    best, best_f1 = None, -1
    for p in config.REPORTS_DIR.glob("*_metrics.json"):
        m = json.loads(p.read_text())
        if m["f1"] > best_f1 and (config.MODELS_DIR / f"{m['model']}.h5").exists():
            best, best_f1 = m["model"], m["f1"]
    if best is None:
        found = sorted(config.MODELS_DIR.glob("*.h5"))
        best = found[0].stem if found else None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--show", type=int, default=15, help="how many random test images to print")
    args = ap.parse_args()

    name = args.model or pick_default_model()
    if name is None:
        raise SystemExit("No trained model found in saved_models/. Run train.py first.")
    names = json.loads((config.REPORTS_DIR / "class_names.json").read_text())
    test_dir = data_utils.get_split_dirs()["test"]
    files = data_utils.list_images(test_dir)
    print(f"Model: {name} | classes: {names} | test images: {len(files)}")

    model = tf.keras.models.load_model(config.MODELS_DIR / f"{name}.h5", compile=False)
    probs = []
    for i in range(0, len(files), 32):
        batch = np.stack([
            np.asarray(Image.open(p).convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR), dtype="float32") / 255.0
            for p in files[i:i + 32]
        ])
        probs.append(model.predict(batch, verbose=0))
    probs = np.concatenate(probs)

    df = pd.DataFrame({
        "File": [p.name for p in files],
        "Actual": [p.parent.name for p in files],
        "Predicted": [names[i] for i in probs.argmax(1)],
        "Confidence": probs.max(1).round(3),
    })
    df["Correct"] = df.Actual == df.Predicted
    df.to_csv(config.REPORTS_DIR / f"test_predictions_{name}.csv", index=False)

    print(f"\nOverall test accuracy: {df.Correct.mean():.2%}  ({int(df.Correct.sum())}/{len(df)})")
    print("\nAccuracy per class:")
    print(df.groupby("Actual").Correct.mean().map("{:.1%}".format).to_string())
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.crosstab(df.Actual, df.Predicted))
    if df.Predicted.nunique() == 1:
        print(f"\nWARNING: the model predicts only '{df.Predicted.iloc[0]}' for every image. "
              "It is under-trained: run a full training (python train.py --models <name>).")

    print(f"\n{min(args.show, len(df))} random test images:")
    sample = df.sample(min(args.show, len(df)), random_state=1).copy()
    sample["Result"] = np.where(sample.Correct, "RIGHT", "WRONG")
    print(sample[["File", "Actual", "Predicted", "Confidence", "Result"]].to_string(index=False))
    print(f"\nFull results saved to reports/test_predictions_{name}.csv")


if __name__ == "__main__":
    main()