"""Grad-CAM: highlights the MRI regions that drove a prediction.

Works on Keras 2 and Keras 3: the model is replayed layer by layer, so no access
to the internal tensors of the nested pretrained backbone is needed.
"""
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image


def _chain(model):
    return [l for l in model.layers if l.__class__.__name__ != "InputLayer"]


def make_gradcam(model, img_batch, class_index=None):
    """Return (heatmap 0-1, class_index, probabilities) for a (1, H, W, 3) batch."""
    chain = _chain(model)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    # Pass 1: find the last layer that outputs a 4-D feature map
    target, h = 0, x
    for i, layer in enumerate(chain):
        h = layer(h)
        if len(h.shape) == 4:
            target = i

    # Pass 2: replay with a tape, split at that layer
    with tf.GradientTape() as tape:
        h = x
        for layer in chain[: target + 1]:
            h = layer(h)
        conv = h
        tape.watch(conv)
        out = conv
        for layer in chain[target + 1:]:
            out = layer(out)
        if class_index is None:
            class_index = int(tf.argmax(out[0]))
        score = out[:, class_index]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_index, out.numpy()[0]


def overlay(image: Image.Image, cam: np.ndarray, alpha=0.45):
    """Blend a jet-coloured heatmap over the original image."""
    base = image.convert("RGB")
    heat = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.BICUBIC)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(base, dtype="float32") / 255.0 + alpha * colored
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
