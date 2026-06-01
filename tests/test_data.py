"""Characterization tests for photoswitch.data."""

import numpy as np
import pandas as pd
import pytest

from photoswitch.data import load_features_and_labels, load_property_data


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
