import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import transform_data

def train_rf_model(X, y, 
                   n_components=0, 
                   use_pca=False, 
                   n_estimators=1519, 
                   max_features=0.086, 
                   min_samples_leaf=2, 
                   test_set_size=0.2, 
                   n_folds=10):
  
  r2_list = []
  rmse_list = []
  mae_list = []

  print('\nBeginning training loop...')

  for i in range(n_folds):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    X_train, X_test, x_scaler, y_train, y_test, y_scaler = transform_data(X_train, X_test, 
                                                                          y_train, y_test,
                                                                          n_components=n_components,
                                                                          use_pca=use_pca)
    
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
    score = r2_score(y_test, y_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_rf))
    mae = mean_absolute_error(y_test, y_rf)
    
    r2_list.append(score)
    rmse_list.append(rmse)
    mae_list.append(mae)
    
  r2_list = np.array(r2_list)
  rmse_list = np.array(rmse_list)
  mae_list = np.array(mae_list)
  print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
  print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
  print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
  
  return regr_rf