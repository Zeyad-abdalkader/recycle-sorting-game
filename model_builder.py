"""
Rebuilds the EfficientNetB0 architecture in plain Python code, so we can
load weights-only (.weights.h5) instead of a full .keras file. This avoids
"Input 0 of layer ... incompatible" errors that happen when the Keras/TF
version used to train the model differs from the version used to run the
app on another machine.

Must match EXACTLY the architecture used during training.
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = 224
NUM_CLASSES = 6


def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> keras.Model:
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,  # no need for imagenet weights here, we load our own trained weights next
        input_shape=(img_size, img_size, 3),
    )

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)
