import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from hpsklearn import HyperoptEstimator, random_forest_regression

from utils import load_features_and_labels, split_and_rescale_data

n_folds = 10
test_set_size = 0.2

n_estimators = 1519

X, y = load_features_and_labels('./processed_data/rdkit_descriptors.csv' ,'e_iso_pi')
X_train, X_test, x_scaler, y_train, y_test, y_scaler = split_and_rescale_data(X, y, split=0.2)

estim = HyperoptEstimator(regressor=random_forest_regression('my_RF'))
estim.fit(X_train, y_train, valid_size=0.1, n_folds=5, cv_shuffle=True)
print(estim.best_model())