"""Deterministic model training."""

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import BATCH_SIZE, CLASSES, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from .model import build_model


@dataclass(frozen=True)
class TrainingResult:
    probabilities: np.ndarray
    history: dict
    seconds: float


def balanced_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


def train_and_predict(train, validation, test_x, use_class_weights, seed, epochs=EPOCHS):
    """Train one model; without a validation set, 20% of the training data is used."""
    train_x, train_y = train
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    model = build_model(train_x.shape[1:], len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    split = {"validation_data": validation} if validation is not None else {"validation_split": 0.2}

    started = time.time()
    history = model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=balanced_class_weights(train_y) if use_class_weights else None,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
        **split,
    )
    seconds = time.time() - started
    probabilities = model.predict(test_x, verbose=0, batch_size=512)
    return TrainingResult(probabilities, history.history, seconds)
