"""Tests for the GP/NN models, which require the optional ``ml`` extra.

The whole module is skipped when TensorFlow/GPflow are not installed, so the
core CI run (which installs only the core dependencies) stays green.
"""

import numpy as np
import pytest

pytest.importorskip("gpflow")
pytest.importorskip("tensorflow")

from photoswitch.models.gaussian_process import Tanimoto, train_gp_model  # noqa: E402
from photoswitch.models.neural_network import dense_model, train_nn_model  # noqa: E402


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(30, 8)).astype(float)  # binary fingerprint-like
    y = X.sum(axis=1) * 10.0 + rng.normal(scale=0.1, size=30)
    return X, y


def test_tanimoto_kernel_is_symmetric_and_diagonal_is_variance():
    import tensorflow as tf

    kernel = Tanimoto()
    # GPflow parameters default to float64; feed matching dtype as the models do.
    X = tf.constant([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=tf.float64)
    K = kernel.K(X).numpy()
    np.testing.assert_allclose(K, K.T, rtol=1e-6)
    # K_diag matches the variance parameter.
    np.testing.assert_allclose(kernel.K_diag(X).numpy(), [1.0, 1.0], rtol=1e-6)


def test_train_gp_model_returns_model_and_scalers(toy_data):
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data
    model, x_scaler, y_scaler = train_gp_model(X, y, n_folds=2, test_set_size=0.3)
    assert isinstance(x_scaler, StandardScaler)
    assert isinstance(y_scaler, StandardScaler)
    mean, var = model.predict_f(x_scaler.transform(X[:2]))
    assert mean.shape == (2, 1)


def test_dense_model_output_shape():
    model = dense_model(8)
    assert model.output_shape == (None, 1)


def test_train_nn_model_returns_model_and_scalers(toy_data):
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data
    model, x_scaler, y_scaler = train_nn_model(
        X, y, epochs=1, batch_size=8, n_folds=2, test_set_size=0.3
    )
    assert isinstance(x_scaler, StandardScaler)
    assert isinstance(y_scaler, StandardScaler)
    preds = model(x_scaler.transform(X[:2])).numpy()
    assert preds.shape == (2, 1)
