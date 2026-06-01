"""Dense feed-forward neural-network regressor (Keras/TensorFlow)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from photoswitch.models.common import cross_validate


def dense_model(input_size: int) -> tf.keras.Model:
    """Build the dense regression network.

    Args:
        input_size: Length of the input feature vector.

    Returns:
        An uncompiled Keras model mapping ``input_size`` features to one output.
    """
    inpts = tf.keras.layers.Input(shape=(input_size,))
    init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    x = inpts
    for units, name in [
        (1024, "layer1"),
        (2048, "layer2"),
        (2048, "layer3"),
        (1024, "layer4"),
        (256, "layer5"),
    ]:
        x = tf.keras.layers.Dense(
            units, activation=tf.nn.leaky_relu, kernel_initializer=init, name=name
        )(x)
    prediction = tf.keras.layers.Dense(
        1, activation=tf.nn.leaky_relu, kernel_initializer=init, name="layer6"
    )(x)
    return tf.keras.models.Model(inputs=inpts, outputs=prediction)


def train_nn_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 8,
    batch_size: int = 8,
    n_components: int = 0,
    use_pca: bool = False,
    test_set_size: float = 0.2,
    n_folds: int = 10,
) -> tuple[tf.keras.Model, StandardScaler, StandardScaler]:
    """Train and cross-validate the dense network.

    Args:
        X: Feature matrix.
        y: Target vector.
        epochs: Training epochs per fold.
        batch_size: Mini-batch size.
        n_components: Principal components to keep when ``use_pca``.
        use_pca: Reduce feature dimensionality with PCA before fitting.
        test_set_size: Held-out fraction per fold.
        n_folds: Number of cross-validation folds.

    Returns:
        ``(model, x_scaler, y_scaler)`` from the final fold.
    """

    def fit_and_predict(X_train, y_train, fold):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
            batch_size=batch_size
        )
        model = dense_model(X_train.shape[1])
        model.compile(
            loss="mae",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["mae"],
            run_eagerly=True,
        )
        model.fit(train_ds, steps_per_epoch=len(train_ds), epochs=epochs, verbose=1)
        return model, lambda X_test: model(X_test).numpy()

    model, x_scaler, y_scaler, _ = cross_validate(
        X,
        y,
        fit_and_predict,
        n_components=n_components,
        use_pca=use_pca,
        test_set_size=test_set_size,
        n_folds=n_folds,
    )
    return model, x_scaler, y_scaler
