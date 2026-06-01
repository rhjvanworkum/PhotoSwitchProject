"""Loading photoswitch property labels and precomputed feature matrices.

The label column mapping follows the original Photoswitch Dataset
(https://github.com/Ryan-Rhys/The-Photoswitch-Dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Maps a short task name to the wavelength/rate column in photoswitches.csv.
TASK_COLUMNS: dict[str, str] = {
    "thermal": "rate of thermal isomerisation from Z-E in s-1",
    "e_iso_pi": "E isomer pi-pi* wavelength in nm",
    "z_iso_pi": "Z isomer pi-pi* wavelength in nm",
    "e_iso_n": "E isomer n-pi* wavelength in nm",
    "z_iso_n": "Z isomer n-pi* wavelength in nm",
}


def load_property_data(df: pd.DataFrame, task: str) -> np.ndarray:
    """Return the target column for ``task`` as a NumPy array.

    Args:
        df: A dataframe loaded from photoswitches.csv.
        task: One of the keys of :data:`TASK_COLUMNS`.

    Returns:
        The selected property values (with NaNs preserved).

    Raises:
        Exception: If ``task`` is not a recognised property name.
    """
    if task not in TASK_COLUMNS:
        raise Exception("Must specify a valid task")
    return df[TASK_COLUMNS[task]].to_numpy()


def load_features_and_labels(
    feature_file: str, smiles_file: str, task: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load a feature matrix and the matching labels, dropping invalid rows.

    Rows whose label is NaN are removed from both the features and the labels.
    The first two columns of ``feature_file`` (the unnamed index and the SMILES
    string) are stripped from the returned feature matrix.

    Args:
        feature_file: CSV of precomputed features (index, SMILES, features...).
        smiles_file: CSV containing the ``SMILES`` column and property columns.
        task: Property name understood by :func:`load_property_data`.

    Returns:
        ``(X, y)`` as NumPy arrays.
    """
    df = pd.read_csv(smiles_file)
    smiles_list = df["SMILES"].to_numpy()

    property_vals = load_property_data(df, task)

    # Drop the index and SMILES columns from the feature file.
    features = pd.read_csv(feature_file).to_numpy()[:, 2:]

    # Remove rows without a valid label.
    invalid_indices = np.argwhere(np.isnan(property_vals))
    smiles_list = np.delete(np.array(smiles_list), invalid_indices)
    property_vals = np.delete(property_vals, invalid_indices)
    features = np.delete(features, invalid_indices, axis=0)

    return features, property_vals
