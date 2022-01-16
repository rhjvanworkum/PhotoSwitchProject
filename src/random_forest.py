import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.utils import transform_data

def train_rf_model(X, y, 
                   n_components=0, 
                   use_pca=False, 
                   n_estimators=1519, 
                   max_features=0.086, 
                   min_samples_leaf=2, 
                   test_set_size=0.2, 
                   n_folds=10):
  """Function to train a random forest model and evaluate it's performance using cross-fold validation

  Args:
      X (np.ndarray): numpy array containing features
      y (np.ndarray): numpy array containing labels
      n_components (int, optional): amount of principal components to keep in the features if using pca. Defaults to 0.
      use_pca (bool, optional): using pca to reduce the dimensionality of the training data. Defaults to False.
      n_estimators (int, optional): amount of estimators used in the RF model. Defaults to 1519.
      max_features (float, optional): hyperparameters of the RF model. Defaults to 0.086.
      min_samples_leaf (int, optional): hyperparameters of the RF model. Defaults to 2.
      test_set_size (float, optional): the ratio between test/train data. Defaults to 0.2.
      n_folds (int, optional): amount of folds for the cross-fold validation. Defaults to 10.

  Returns:
      model, x_scaler, y_scaler
  """
  
  r2_list = []
  rmse_list = []
  mae_list = []

  print('\nBeginning training loop...')

  for i in range(n_folds):
    # load and tranform the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    X_train, X_test, x_scaler, y_train, y_test, y_scaler = transform_data(X_train, X_test, 
                                                                          y_train, y_test,
                                                                          n_components=n_components,
                                                                          use_pca=use_pca)
    
    # initiate random forest regressor
    regr_rf = RandomForestRegressor(n_estimators=n_estimators, 
                                    random_state=i, 
                                    max_features=max_features, 
                                    bootstrap=False, 
                                    min_samples_leaf=min_samples_leaf)
  
    regr_rf.fit(X_train, y_train.ravel())
    
    # Predict on new data
    y_rf = regr_rf.predict(X_test).reshape(-1, 1)
    y_rf = y_scaler.inverse_transform(y_rf)
    y_test = y_scaler.inverse_transform(y_test)
    
    # compute the performance metrices
    r2_list.append(r2_score(y_test, y_rf))
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_rf)))
    mae_list.append(mean_absolute_error(y_test, y_rf))
    
  # print out the performance metrics
  r2_list = np.array(r2_list)
  rmse_list = np.array(rmse_list)
  mae_list = np.array(mae_list)
  print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
  print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
  print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
  
  return regr_rf, x_scaler, y_scaler