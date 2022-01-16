import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf

from src.utils import transform_data

# this is a tensorflow callback in order to lower the learning rate during training
reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.95,
    patience=3,
    verbose=1,
    mode='min',
    min_delta=0.0001,
    cooldown=2,
    min_lr=1e-5
)

# this is a tensorflow callback in order to save intermediate versions of the trained model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='../runs/saved-model-{epoch:02d}-{val_acc:.2f}.h5',
    monitor='val_acc',
    mode='max',
    save_best_only=False
)

def dense_model(input_size):
  """Function to generate the keras implemented dense neural network

  Args:
      input_size (int): length of the input feature vector

  Returns:
      tf.keras.Model: a keras Model that can be trained 
  """
  inpts = tf.keras.layers.Input(shape=(input_size, ))
  x = tf.keras.layers.Dense(1024,
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                            name="layer1")(inpts)
  x = tf.keras.layers.Dense(2048, 
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                            name="layer2")(x)
  x = tf.keras.layers.Dense(2048, 
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                            name="layer3")(x)
  x = tf.keras.layers.Dense(1024, 
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                            name="layer4")(x)
  x = tf.keras.layers.Dense(256, 
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                            name="layer5")(x)
  prediction = tf.keras.layers.Dense(1, 
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                            name="layer6")(x)
  
  model = tf.keras.models.Model(inputs=inpts, outputs=prediction)
  
  return model

def train_nn_model(X, y, 
                   epochs=8,
                   batch_size=8,
                   n_components=0, 
                   use_pca=False, 
                   test_set_size=0.2, 
                   n_folds=10):
  """Function to train a dense neural network model and evaluate it's performance using cross-fold validation

  Args:
      X (np.ndarray): numpy array containing features
      y (np.ndarray): numpy array containing labels
      epochs: (int, optional): amount of epochs to train for. Defaults to 8.
      batch_size: (int, optional): amount of training samples to put in 1  training batch. Defaults to 8.
      n_components (int, optional): amount of principal components to keep in the features if using pca. Defaults to 0.
      use_pca (bool, optional): using pca to reduce the dimensionality of the training data. Defaults to False.
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
    
    # load the data inside a tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size=batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size=batch_size)
    
    # initiate random forest regressor
    model = dense_model(X_train.shape[1])
    
    # compile the model with an Adam optimizer and MAE loss
    model.compile(loss='mae',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae'],
                  run_eagerly=True)
    
    # fit the model to the data
    hist = model.fit(
      train_ds,
      steps_per_epoch=len(train_ds),
      validation_data=test_ds,
      validation_steps=len(test_ds),
      epochs=epochs,
      verbose=1)
    
    # Predict on new data
    y_rf = model(X_test).numpy().reshape(-1, 1)
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
  
  return model, x_scaler, y_scaler