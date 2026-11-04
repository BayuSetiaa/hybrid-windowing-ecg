"""One-dimensional CNN with squeeze-and-excitation blocks and a residual connection."""

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
    Multiply,
)
from tensorflow.keras.models import Model


def squeeze_excitation(inputs, ratio=8):
    channels = inputs.shape[-1]
    weights = GlobalAveragePooling1D()(inputs)
    weights = Dense(channels // ratio, activation="relu")(weights)
    weights = Dense(channels, activation="sigmoid")(weights)
    return Multiply()([inputs, weights])


def residual_block(inputs, filters, kernel_size):
    x = Conv1D(filters, kernel_size, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(Add()([x, inputs]))


def convolution_block(inputs, filters, kernel_size):
    x = Conv1D(filters, kernel_size, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return squeeze_excitation(x)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = convolution_block(inputs, 32, 7)
    x = MaxPooling1D(2)(x)
    x = convolution_block(x, 64, 5)
    x = MaxPooling1D(2)(x)
    x = convolution_block(x, 128, 3)
    x = residual_block(x, 128, 3)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)
