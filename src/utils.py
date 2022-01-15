import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_property_data(df, task):
    """
    Load data corresponding to the property prediction task.
    :return: property_vals
    
    Taken from: https://github.com/Ryan-Rhys/The-Photoswitch-Dataset
    """
    if task == 'thermal':
        # Load the SMILES as x-values and the rate of thermal isomerisation as the y-values
        property_vals = df['rate of thermal isomerisation from Z-E in s-1'].to_numpy()
    elif task == 'e_iso_pi':
        #     Load the SMILES as x-values and the E isomer pi-pi* wavelength in nm as the y-values.
        #     108 molecules with valid experimental values as of 11 May 2020.
        property_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()
    elif task == 'z_iso_pi':
        #     Load the SMILES as x-values and the Z isomer pi-pi* wavelength in nm as the y-values.
        #     84 valid molecules for this property as of 11 May 2020.
        property_vals = df['Z isomer pi-pi* wavelength in nm'].to_numpy()
    elif task == 'e_iso_n':
        #     Load the SMILES as x-values and the E isomer n-pi* wavelength in nm as the y-values.
        #     96 valid molecules for this property as of 9 May 2020.
        property_vals = df['E isomer n-pi* wavelength in nm'].to_numpy()
    elif task == 'z_iso_n':
        #     Load the SMILES as x-values and the Z isomer n-pi* wavelength in nm as the y-values.
        #     93 valid molecules with this property as of 9 May 2020
        #     114 valid molecules with this property as of 16 May 2020
        property_vals = df['Z isomer n-pi* wavelength in nm'].to_numpy()
    else:
        raise Exception('Must specify a valid task')

    return property_vals
  
def transform_data(X_train, X_test, y_train, y_test, n_components=0, use_pca=False):
    """Transforming the dataset using sklearn standard scaler & pca optionally

    Args:
        X_train ([np.array]): [training features]
        X_test ([np.array]): [test features]
        y_train ([np.array]): [training labels]
        y_test ([np.array]): [test labels]
        n_components (int, optional): [how many principal components to use as input features]. Defaults to 0.
        use_pca (bool, optional): [using pca to reduce the amount of input features fed to the model]. Defaults to False.

    Returns:
        X_train_scaled, X_test_scaled, x_scaler, y_train_scaled, y_test_scaled, y_scaler
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
      print('(PCA) Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
      X_test_scaled = pca.transform(X_test)

    return X_train_scaled, X_test_scaled, x_scaler, y_train_scaled, y_test_scaled, y_scaler
  
def split_and_rescale_data(X, y, split):
    """Split the dataset and rescale the features using standard scaling

    Args:
        X ([np.array]): [Features]
        y ([np.array]): [labels]
        split ([int]): [split fraction between train/test data]

    Returns:
        X_train_scaled, X_test_scaled, x_scaler, y_train_scaled, y_test_scaled, y_scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return transform_data(X_train, X_test, y_train, y_test)
  
def load_features_and_labels(feature_file, smiles_file, task):
    """Loads the features and labels from the respective input files

    Args:
        feature_file ([string]): [file containing features]
        smiles_file ([string]): [file containing the initial dataset]
        task ([string]): [name of the property to extract]

    Returns:
        X, y: np.array containing features & np.array containing the labels
    """
    
    # load the smiles list
    df = pd.read_csv(smiles_file)
    smiles_list = df['SMILES'].to_numpy()
    
    # load the property values (the labels)
    property_vals = load_property_data(df, task)
    
    # remove index and smiles string
    features = pd.read_csv(feature_file).to_numpy()[:, 2:] 
    
    # remove unvalid indices
    invalid_indices = np.argwhere(np.isnan(property_vals))
    smiles_list = np.delete(np.array(smiles_list), invalid_indices)
    property_vals = np.delete(property_vals, invalid_indices)
    features = np.delete(features, invalid_indices, axis=0)
    
    X = features
    y = property_vals
    
    return X, y

def plot_pca(file, task):
    """Use PCA to make a plot of the first 2 principal components

    Args:
        file ([string]): [the path to the file containing the features to apply pca on]
        task ([string]): [the task for which to extract the right data]
    """
    df = pd.read_csv('../raw_data/photoswitches.csv')
    labels = load_property_data(df, task)
    invalid_indices = np.argwhere(np.isnan(labels))
    
    labels = np.delete(labels, invalid_indices)
    
    features = pd.read_csv(file).to_numpy()[:, 2:]
    features = np.delete(features, invalid_indices, axis=0)

    pca = PCA(n_components = 2)
    components = pca.fit_transform(features)
    
    print('explained variance: ', pca.explained_variance_ratio_)
    plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='bwr')
    plt.show()