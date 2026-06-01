"""Plotting helpers for exploring the feature space."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from photoswitch.data import load_property_data


def plot_pca(
    feature_file: str,
    task: str,
    smiles_file: str = "raw_data/photoswitches.csv",
) -> None:
    """Scatter the first two principal components of a feature file.

    Points are coloured by the property values for ``task``.

    Args:
        feature_file: CSV of features (index, SMILES, features...).
        task: Property name understood by :func:`photoswitch.data.load_property_data`.
        smiles_file: CSV with the ``SMILES`` column and property columns. Defaults
            to the repository's ``raw_data/photoswitches.csv`` (relative to the
            working directory). Previously this path was hard-coded and pointed at
            a non-existent location; it is now an explicit, overridable argument.
    """
    df = pd.read_csv(smiles_file)
    labels = load_property_data(df, task)
    invalid_indices = np.argwhere(np.isnan(labels))

    labels = np.delete(labels, invalid_indices)

    features = pd.read_csv(feature_file).to_numpy()[:, 2:]
    features = np.delete(features, invalid_indices, axis=0)

    pca = PCA(n_components=2)
    components = pca.fit_transform(features)

    print("explained variance: ", pca.explained_variance_ratio_)
    plt.scatter(components[:, 0], components[:, 1], c=labels, cmap="bwr")
    plt.show()
