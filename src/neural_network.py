import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf

from src.utils import transform_data

def dense_model(input_size):
  inpts = tf.keras.layers.Input(shape=(input_size, ))
  x = tf.keras.layers.Dense(1024,
                            activation=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=22) ,
                            normalizer_fn=tf.batch_norm,
                            name="layer1")(inpts)
  x = tf.keras.layers.Dense(2048, 
                            activation=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=22) ,
                            normalizer_fn=tf.batch_norm,
                            name="layer2")(x)
  x = tf.keras.layers.Dense(2048, 
                            activation=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=22) ,
                            normalizer_fn=tf.batch_norm,
                            name="layer3")(x)
  x = tf.keras.layers.Dense(1024, 
                            activation=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=22) ,
                            normalizer_fn=tf.batch_norm,
                            name="layer4")(x)
  x = tf.keras.layers.Dense(256, 
                            activation=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=22) ,
                            normalizer_fn=tf.batch_norm,
                            name="layer5")(x)
  prediction = tf.keras.layers.Dense(1, 
                            activation=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=22) ,
                            normalizer_fn=tf.batch_norm,
                            name="layer6")(x)
  
  model = tf.keras.models.Model(inputs=inpts, outputs=prediction)
  
  return model

def train_nn_model(X, y, 
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
    # load and tranform the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    X_train, X_test, x_scaler, y_train, y_test, y_scaler = transform_data(X_train, X_test, 
                                                                          y_train, y_test,
                                                                          n_components=n_components,
                                                                          use_pca=use_pca)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # initiate random forest regressor
    model = dense_model(X_train.shape[1])
    
    model.compile(loss=tf.keras.losses.huber_lossy,
                  optimizer=tf.keras.optimizers.Adam())
    
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

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='../runs/' + i + '/saved-model-{epoch:02d}-{val_acc:.2f}.h5',
        monitor='val_acc',
        mode='max',
        save_best_only=False
    )
    
    epochs = 20
    batch_size = 32
    
    hist = model.fit(
      train_ds,
      steps_per_epoch=int(len(train_ds) / batch_size),
      validation_data=test_ds,
      validation_steps=int(len(test_ds) / batch_size),
      epochs=epochs,
      verbose=1,
      callbacks=[reduce_lr_on_plat, model_checkpoint])
    
    # Predict on new data
    y_rf = model(X_test).reshape(-1, 1)
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