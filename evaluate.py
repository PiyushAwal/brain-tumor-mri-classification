"""Evaluate trained models on the test set and build the comparison table.

Usage:  python evaluate.py            # re-evaluates every .h5 in saved_models/
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

import config


def evaluate_model(model, test_ds, class_names, model_path, name):
    """Return a metrics dict (accuracy, P/R/F1, confusion matrix, latency, size)."""
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    for x, _ in test_ds.take(1):  # warm-up so latency is not dominated by graph tracing
        model.predict(x, verbose=0)
    t0 = time.perf_counter()
    probs = model.predict(test_ds, verbose=0)
    elapsed = time.perf_counter() - t0
    y_pred = probs.argmax(axis=1)

    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "model": name,
        "class_names": list(class_names),
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "latency_ms": float(elapsed / len(y_true) * 1000),
        "params": int(model.count_params()),
        "size_mb": float(os.path.getsize(model_path) / 1e6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(class_names)
        },
    }


def save_metrics(metrics):
    path = config.REPORTS_DIR / f"{metrics['model']}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def build_comparison():
    """Collect all *_metrics.json into reports/comparison.csv (sorted by macro F1)."""
    rows = []
    for p in sorted(config.REPORTS_DIR.glob("*_metrics.json")):
        m = json.loads(p.read_text())
        rows.append({
            "model": m["model"], "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "latency_ms": m["latency_ms"],
            "params_millions": m["params"] / 1e6, "size_mb": m["size_mb"],
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(config.REPORTS_DIR / "comparison.csv", index=False)
    return df


if __name__ == "__main__":
    import tensorflow as tf
    import data_utils

    _, _, test_ds, class_names, _ = data_utils.load_datasets(augment=False)
    for path in sorted(config.MODELS_DIR.glob("*.h5")):
        print(f"Evaluating {path.stem} ...")
        model = tf.keras.models.load_model(path, compile=False)
        save_metrics(evaluate_model(model, test_ds, class_names, path, path.stem))
    print(build_comparison())
