"""Random-forest regression model with cross-validated evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from photoswitch.models.common import cross_validate


def train_rf_model(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 0,
    use_pca: bool = False,
    n_estimators: int = 1519,
    max_features: float = 0.086,
    min_samples_leaf: int = 2,
    test_set_size: float = 0.2,
    n_folds: int = 10,
) -> tuple[RandomForestRegressor, StandardScaler, StandardScaler]:
    """Train and cross-validate a random-forest regressor.

    Args:
        X: Feature matrix.
        y: Target vector.
        n_components: Principal components to keep when ``use_pca``.
        use_pca: Reduce feature dimensionality with PCA before fitting.
        n_estimators: Number of trees in the forest.
        max_features: Fraction of features considered per split.
        min_samples_leaf: Minimum samples per leaf.
        test_set_size: Held-out fraction per fold.
        n_folds: Number of cross-validation folds.

    Returns:
        ``(model, x_scaler, y_scaler)`` from the final fold.
    """

    def fit_and_predict(X_train, y_train, fold):
        regr = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=fold,
            max_features=max_features,
            bootstrap=False,
            min_samples_leaf=min_samples_leaf,
        )
        regr.fit(X_train, y_train.ravel())
        return regr, lambda X_test: regr.predict(X_test)

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
