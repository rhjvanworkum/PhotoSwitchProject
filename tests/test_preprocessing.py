"""Characterization tests for photoswitch.preprocessing."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from photoswitch.preprocessing import split_and_rescale_data, transform_data


class TestTransformData:
    def test_scaling_is_standard_and_returns_scalers(self):
        X_train = np.array([[0.0], [2.0], [4.0]])
        X_test = np.array([[6.0]])
        y_train = np.array([[10.0], [20.0], [30.0]])
        y_test = np.array([[40.0]])
        Xtr, Xte, xs, ytr, yte, ys = transform_data(X_train, X_test, y_train, y_test)
        assert isinstance(xs, StandardScaler)
        assert isinstance(ys, StandardScaler)
        # train mean ~0 after standardization
        assert Xtr.mean() == pytest.approx(0.0, abs=1e-9)
        # inverse transform round-trips
        np.testing.assert_allclose(ys.inverse_transform(ytr), y_train)

    def test_pca_reduces_feature_dimension(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(10, 5))
        X_test = rng.normal(size=(3, 5))
        y_train = rng.normal(size=(10, 1))
        y_test = rng.normal(size=(3, 1))
        Xtr, Xte, *_ = transform_data(
            X_train, X_test, y_train, y_test, n_components=2, use_pca=True
        )
        assert Xtr.shape == (10, 2)
        assert Xte.shape == (3, 2)


class TestSplitAndRescaleData:
    def test_returns_six_outputs_and_is_deterministic(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10, dtype=float)
        out1 = split_and_rescale_data(X, y, 0.2)
        out2 = split_and_rescale_data(X, y, 0.2)
        assert len(out1) == 6
        # random_state is fixed at 1 inside the function, so splits are identical.
        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_array_equal(out1[1], out2[1])
