"""Characterization tests for photoswitch.utils.

These lock in the *current* behavior of the deterministic data-handling helpers
before the Phase 4 refactor reorganizes them.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from photoswitch.utils import (
    load_features_and_labels,
    load_property_data,
    split_and_rescale_data,
    transform_data,
)


class TestLoadPropertyData:
    def test_e_iso_pi_returns_column_as_array(self):
        df = pd.DataFrame({"E isomer pi-pi* wavelength in nm": [1.0, 2.0, 3.0]})
        result = load_property_data(df, "e_iso_pi")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        ("task", "column"),
        [
            ("thermal", "rate of thermal isomerisation from Z-E in s-1"),
            ("e_iso_pi", "E isomer pi-pi* wavelength in nm"),
            ("z_iso_pi", "Z isomer pi-pi* wavelength in nm"),
            ("e_iso_n", "E isomer n-pi* wavelength in nm"),
            ("z_iso_n", "Z isomer n-pi* wavelength in nm"),
        ],
    )
    def test_each_task_maps_to_its_column(self, task, column):
        df = pd.DataFrame({column: [4.0, 5.0]})
        np.testing.assert_array_equal(load_property_data(df, task), [4.0, 5.0])

    def test_invalid_task_raises(self):
        with pytest.raises(Exception, match="valid task"):
            load_property_data(pd.DataFrame(), "not_a_task")


class TestLoadFeaturesAndLabels:
    def test_drops_rows_with_nan_labels(self, feature_csv, photoswitch_csv):
        # Row index 2 has a NaN E-iso label and must be dropped from X and y.
        X, y = load_features_and_labels(str(feature_csv), str(photoswitch_csv), "e_iso_pi")
        assert X.shape == (4, 3)  # 5 rows - 1 NaN, 3 feature columns (index+SMILES stripped)
        assert y.shape == (4,)
        assert not np.isnan(y).any()
        np.testing.assert_array_equal(y, [310.0, 320.0, 305.0, 330.0])

    def test_features_strip_index_and_smiles_columns(self, feature_csv, photoswitch_csv):
        X, _ = load_features_and_labels(str(feature_csv), str(photoswitch_csv), "e_iso_pi")
        # First surviving row's features are f0,f1,f2 of SMILES[0].
        np.testing.assert_array_equal(X[0], [1.0, 0.1, 10.0])


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
