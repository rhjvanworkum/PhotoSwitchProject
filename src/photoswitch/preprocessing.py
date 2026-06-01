"""Feature/target scaling and train-test splitting helpers."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def transform_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 0,
    use_pca: bool = False,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, np.ndarray, np.ndarray, StandardScaler]:
    """Standard-scale features and targets, optionally reducing features with PCA.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        n_components: Number of principal components to keep when ``use_pca``.
        use_pca: Whether to reduce feature dimensionality with PCA.

    Returns:
        ``(X_train_scaled, X_test_scaled, x_scaler, y_train_scaled, y_test_scaled,
        y_scaler)``.
    """
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train)
        print("(PCA) Fraction of variance retained is: " + str(sum(pca.explained_variance_ratio_)))
        X_test_scaled = pca.transform(X_test)

    return X_train_scaled, X_test_scaled, x_scaler, y_train_scaled, y_test_scaled, y_scaler


def split_and_rescale_data(
    X: np.ndarray, y: np.ndarray, split: float
) -> tuple[np.ndarray, np.ndarray, StandardScaler, np.ndarray, np.ndarray, StandardScaler]:
    """Split into train/test (fixed ``random_state=1``) then standard-scale.

    Args:
        X: Features.
        y: Targets.
        split: Test-set fraction passed to ``train_test_split``.

    Returns:
        Same tuple as :func:`transform_data`.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return transform_data(X_train, X_test, y_train, y_test)
