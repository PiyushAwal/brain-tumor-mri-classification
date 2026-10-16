"""Model definitions: custom CNN + ResNet50 / MobileNetV2 / InceptionV3 / EfficientNetB0.

All models take 0-1 normalised RGB input. Backbone-specific preprocessing is baked
into a serialisable Rescaling layer so the saved .h5 files work with plain 0-1 images.
"""
import tensorflow as tf
from tensorflow.keras import applications, layers, models, regularizers

import config

INPUT_SHAPE = (*config.IMG_SIZE, 3)

# (constructor, Rescaling kwargs mapping 0-1 input -> the range the backbone expects)
BACKBONES = {
    "resnet50": (applications.ResNet50, dict(scale=255.0, offset=-115.0)),   # ~ImageNet mean shift
    "mobilenetv2": (applications.MobileNetV2, dict(scale=2.0, offset=-1.0)),  # [-1, 1]
    "inceptionv3": (applications.InceptionV3, dict(scale=2.0, offset=-1.0)),  # [-1, 1]
    "efficientnetb0": (applications.EfficientNetB0, dict(scale=255.0, offset=0.0)),  # 0-255
}


def _head(x, num_classes):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_custom_cnn(num_classes):
    """4 conv blocks (2x Conv-BN-ReLU + MaxPool + Dropout) followed by a dense head."""
    inputs = layers.Input(INPUT_SHAPE)
    x = inputs
    for filters in (32, 64, 128, 256):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    return models.Model(inputs, _head(x, num_classes), name="custom_cnn_clf")


def build_transfer_model(name, num_classes):
    """ImageNet backbone (frozen) + new classification head."""
    ctor, pre = BACKBONES[name]
    inputs = layers.Input(INPUT_SHAPE)
    x = layers.Rescaling(**pre, name="backbone_preprocess")(inputs)
    base = ctor(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm in inference mode
    return models.Model(inputs, _head(x, num_classes), name=f"{name}_clf")


def get_backbone(model):
    """Return the nested pretrained backbone (a Model used as a layer)."""
    return next(l for l in model.layers if isinstance(l, tf.keras.Model))


def unfreeze_top(base, n_layers=config.UNFREEZE_LAYERS):
    """Unfreeze the last n layers for fine-tuning; BatchNorm stays frozen."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def build_model(name, num_classes):
    return build_custom_cnn(num_classes) if name == "custom_cnn" else build_transfer_model(name, num_classes)
