"""Characterization test for photoswitch.random_forest.train_rf_model.

Locks the (model, x_scaler, y_scaler) return contract and the cross-validation
loop's shape before the shared CV loop is extracted in Phase 4. Uses a tiny
forest so the test stays fast.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from photoswitch.random_forest import train_rf_model


def test_returns_model_and_scalers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.1, size=40)

    model, x_scaler, y_scaler = train_rf_model(
        X, y, n_estimators=5, n_folds=2, test_set_size=0.25
    )

    assert isinstance(model, RandomForestRegressor)
    assert isinstance(x_scaler, StandardScaler)
    assert isinstance(y_scaler, StandardScaler)
    # The model was fit, so it can predict on scaled features.
    preds = model.predict(x_scaler.transform(X[:3]))
    assert preds.shape == (3,)
