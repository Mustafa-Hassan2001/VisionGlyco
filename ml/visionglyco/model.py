"""EfficientNet transfer-learning model."""

import keras
from keras import layers

from .config import IMAGE_SIZE, TASK_CLASSIFY

BACKBONES = {
    "efficientnet_b0": keras.applications.EfficientNetB0,
    "efficientnet_b3": keras.applications.EfficientNetB3,
}


def build_model(task, image_size=IMAGE_SIZE, backbone="efficientnet_b0", weights="imagenet",
                target_mean=0.0, target_std=1.0, dropout=0.3):
    """Pretrained backbone + small head.

    Input: float32 RGB in 0-255, shape (image_size, image_size, 3).
    Output: P(positive) for classification, or the target in its own units for
    regression (the head predicts a standardised value that is rescaled here, so
    the exported model needs no post-processing on the phone).
    """
    base = BACKBONES[backbone](include_top=False, weights=weights,
                               input_shape=(image_size, image_size, 3))
    base.trainable = False

    inputs = keras.Input((image_size, image_size, 3), name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    if task == TASK_CLASSIFY:
        outputs = layers.Dense(1, activation="sigmoid", name="probability")(x)
    else:
        x = layers.Dense(1, name="standardised")(x)
        outputs = layers.Rescaling(scale=target_std, offset=target_mean, name="value")(x)
    return keras.Model(inputs, outputs, name=f"visionglyco_{task}"), base


def compile_model(model, task, learning_rate):
    optimizer = keras.optimizers.Adam(learning_rate)
    if task == TASK_CLASSIFY:
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc"),
                               keras.metrics.BinaryAccuracy(name="accuracy")])
    else:
        model.compile(optimizer=optimizer, loss=keras.losses.Huber(),
                      metrics=[keras.metrics.MeanAbsoluteError(name="mae")])


def unfreeze_top(base, n_layers):
    """Unfreeze the last n_layers of the backbone for fine-tuning.

    BatchNormalization layers stay frozen: updating their statistics on a small
    medical dataset usually destroys the pretrained features.
    """
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
