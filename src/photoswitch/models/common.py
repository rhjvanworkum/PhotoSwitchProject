"""Shared cross-validation scaffolding for the property-prediction models.

The random-forest, Gaussian-process and neural-network trainers all repeated the
same loop: split the data, standard-scale it, fit a model, predict on the held
out fold, inverse-transform and accumulate R^2 / RMSE / MAE. That loop lives
here once; each model supplies only a ``fit_and_predict`` callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from photoswitch.preprocessing import transform_data

# A predict function maps scaled test features to scaled predictions (n, 1).
PredictFn = Callable[[np.ndarray], np.ndarray]
# A fit_and_predict callback fits a model on one fold and returns the fitted
# model together with its predict function. It receives the fold index so a
# model can seed itself reproducibly (as the random forest does).
FitAndPredict = Callable[[np.ndarray, np.ndarray, int], "tuple[Any, PredictFn]"]


@dataclass
class CVMetrics:
    """Per-fold regression metrics gathered across a cross-validation run."""

    r2: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray

    def summary(self) -> str:
        def line(name: str, values: np.ndarray) -> str:
            stderr = np.std(values) / np.sqrt(len(values))
            return f"mean {name}: {np.mean(values):.4f} +- {stderr:.4f}"

        return "\n".join([line("R^2", self.r2), line("RMSE", self.rmse), line("MAE", self.mae)])


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    fit_and_predict: FitAndPredict,
    *,
    n_components: int = 0,
    use_pca: bool = False,
    test_set_size: float = 0.2,
    n_folds: int = 10,
    verbose: bool = True,
) -> tuple[Any, StandardScaler, StandardScaler, CVMetrics]:
    """Run a repeated train/test split evaluation of a regression model.

    Args:
        X: Feature matrix.
        y: Target vector.
        fit_and_predict: Callback ``(X_train, y_train, fold) -> (model, predict)``
            where ``predict(X_test)`` returns scaled predictions.
        n_components: Number of principal components to keep when ``use_pca``.
        use_pca: Whether to reduce dimensionality with PCA before fitting.
        test_set_size: Held-out fraction for each fold.
        n_folds: Number of train/test splits (each with ``random_state=fold``).
        verbose: Print the metric summary, matching the original trainers.

    Returns:
        The model fitted on the final fold, the fitted feature and target
        scalers, and the collected :class:`CVMetrics`.
    """
    r2_list: list[float] = []
    rmse_list: list[float] = []
    mae_list: list[float] = []

    if verbose:
        print("\nBeginning training loop...")

    model: Any = None
    x_scaler: StandardScaler | None = None
    y_scaler: StandardScaler | None = None

    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_set_size, random_state=fold
        )
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_train, X_test, x_scaler, y_train, y_test, y_scaler = transform_data(
            X_train, X_test, y_train, y_test, n_components=n_components, use_pca=use_pca
        )

        model, predict = fit_and_predict(X_train, y_train, fold)

        y_pred = predict(X_test).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred)
        y_true = y_scaler.inverse_transform(y_test)

        r2_list.append(r2_score(y_true, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_list.append(mean_absolute_error(y_true, y_pred))

    metrics = CVMetrics(r2=np.array(r2_list), rmse=np.array(rmse_list), mae=np.array(mae_list))

    if verbose:
        print("\n" + metrics.summary() + "\n")

    if x_scaler is None or y_scaler is None:
        raise ValueError("n_folds must be >= 1")

    return model, x_scaler, y_scaler, metrics
