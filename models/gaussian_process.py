from utils import transform_data


import gpflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf

class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self, **kwargs):
        """
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError("Unknown keyword argument:", kwarg)
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        cross_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * cross_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

def train_gp_model(X, y, n_components=0, use_pca=False, test_set_size=0.2, n_folds=10):
  
  r2_list = []
  rmse_list = []
  mae_list = []
  
  _, _, _, y_test = train_test_split(X, y, test_size=test_set_size)  # To get test set size
  n_test = len(y_test)
  
  rmse_confidence_list = np.zeros((n_folds, n_test))
  mae_confidence_list = np.zeros((n_folds, n_test))

  print('\nBeginning training loop...')

  for i in range(n_folds):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    print(X_train.shape, y_train.shape)
    X_train, X_test, x_scaler, y_train, y_test, y_scaler = transform_data(X_train, X_test, 
                                                                          y_train, y_test,
                                                                          n_components=n_components,
                                                                          use_pca=use_pca)
    
    model = gpflow.models.GPR(data=(X_train, y_train), mean_function=gpflow.mean_functions.Constant(np.mean(y_train)), kernel=Tanimoto(), noise_variance=1)
    optimizer = gpflow.optimizers.Scipy()
    def closure():
        return -model.log_marginal_likelihood()
    optimizer.minimize(closure, model.trainable_variables, options=dict(maxiter=10000))
    # gpflow.utilities.print_summary(model)
    
    y_pred, y_var = model.predict_f(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)
    
    ranked_confidence_list = np.argsort(y_var, axis=0).flatten()
    
    for k in range(len(y_test)):
      # Construct the RMSE error for each level of confidence
      conf = ranked_confidence_list[0:k+1]
      rmse = np.sqrt(mean_squared_error(y_test[conf], y_pred[conf]))
      rmse_confidence_list[i, k] = rmse
      # Construct the MAE error for each level of confidence
      mae = mean_absolute_error(y_test[conf], y_pred[conf])
      mae_confidence_list[i, k] = mae
      
    # Output Standardised RMSE and RMSE on Train Set
    # y_pred_train, _ = model.predict_f(X_train)
    # train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
    # train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train)))

    score = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2_list.append(score)
    rmse_list.append(rmse)
    mae_list.append(mae)

  r2_list = np.array(r2_list)
  rmse_list = np.array(rmse_list)
  mae_list = np.array(mae_list)

  print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
  print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
  print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
  
  return model, x_scaler, y_scaler
  
def get_gp_data(X, y, smiles, n_components=0, use_pca=False, test_set_size=0.2):
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=test_set_size, random_state=0)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    X_train, X_test, x_scaler, y_train, y_test, y_scaler = transform_data(X_train, X_test, 
                                                                          y_train, y_test,
                                                                          n_components=n_components,
                                                                          use_pca=use_pca)
    
    model = gpflow.models.GPR(data=(X_train, y_train), mean_function=gpflow.mean_functions.Constant(np.mean(y_train)), kernel=Tanimoto(), noise_variance=1)
    optimizer = gpflow.optimizers.Scipy()
    def closure():
        return -model.log_marginal_likelihood()
    optimizer.minimize(closure, model.trainable_variables, options=dict(maxiter=10000))
    # gpflow.utilities.print_summary(model)
    
    y_pred, y_var = model.predict_f(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)
    
    y_train = y_scaler.inverse_transform(y_train)
    
    # NOTE: X_train/X_test is still in transformed values
    return X_test, y_test, smiles_test, y_pred, y_var
    