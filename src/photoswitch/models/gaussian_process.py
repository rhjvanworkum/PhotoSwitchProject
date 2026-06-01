"""Gaussian-process regression with the Tanimoto molecular-similarity kernel.

The Tanimoto kernel is taken from the Photoswitch Dataset project
(https://github.com/Ryan-Rhys/The-Photoswitch-Dataset).
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
from sklearn.preprocessing import StandardScaler

from photoswitch.models.common import cross_validate


class Tanimoto(gpflow.kernels.Kernel):
    """Tanimoto similarity kernel for molecular fingerprints/descriptors."""

    def __init__(self, **kwargs):
        """
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError("Unknown keyword argument:", kwarg)
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        cross_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of X and X2

        # Analogue of denominator in Tanimoto formula
        denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * cross_product / denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


def train_gp_model(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 0,
    use_pca: bool = False,
    test_set_size: float = 0.2,
    n_folds: int = 10,
) -> tuple[gpflow.models.GPR, StandardScaler, StandardScaler]:
    """Train and cross-validate a GP regressor with the Tanimoto kernel.

    Args:
        X: Feature matrix.
        y: Target vector.
        n_components: Principal components to keep when ``use_pca``.
        use_pca: Reduce feature dimensionality with PCA before fitting.
        test_set_size: Held-out fraction per fold.
        n_folds: Number of cross-validation folds.

    Returns:
        ``(model, x_scaler, y_scaler)`` from the final fold.
    """

    def fit_and_predict(X_train, y_train, fold):
        model = gpflow.models.GPR(
            data=(X_train, y_train),
            mean_function=gpflow.mean_functions.Constant(np.mean(y_train)),
            kernel=Tanimoto(),
            noise_variance=1,
        )
        optimizer = gpflow.optimizers.Scipy()
        # model.training_loss is the negative log marginal likelihood; this
        # replaces the removed GPflow API `-model.log_marginal_likelihood()`.
        optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options=dict(maxiter=10000),
        )

        def predict(X_test):
            mean, _ = model.predict_f(X_test)
            return mean.numpy()

        return model, predict

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
