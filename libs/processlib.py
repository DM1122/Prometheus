import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler


def preprocessor(data, label, dropzeros, log):
    # Drop rows w/ zero in label col
    if (dropzeros):
        data[label].replace(0, pd.np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)
    
    # Calculate log element-wise
    if (log):
        # data.loc[(data != 0).any(axis=1)].apply(np.log)
        for col in data:
            if ((data[col] != 0).all(axis=0)):
                data[col] = data[col].apply(np.log)
    
    return data


def labeller(data, label, shift):
    data['Label'] = data[label].shift(-shift)       # create label col
    data.dropna(inplace=True)       # drop gap created by shift length
    data_features = data.drop(columns='Label')      # create features df from all but labels col
    data_labels = data['Label']     # create labels df from labels col

    return data_features, data_labels


def splitter(data_features, data_labels, split_valid, split_test):

    # Convert dataset to numpy arrays
    data_features = np.array(data_features)
    data_labels = np.array(data_labels)

    # Calculate train split from remaining percentage
    split_train = 1 - (split_valid + split_test)

    # Calculate num of observations for each slice
    n_train = int(len(data_features) * split_train)
    n_valid = int(len(data_features) * split_valid)
    n_test = int(len(data_features) * split_test)

    # Slice dataset into train, validate, test
    X_train = data_features[0:n_train]
    y_train = data_labels[0:n_train]

    X_valid = data_features[n_train:n_train+n_valid]
    y_valid = data_labels[n_train:n_train+n_valid]

    X_test = data_features[n_train+n_valid:n_train+n_valid+n_test]
    y_test = data_labels[n_train+n_valid:n_train+n_valid+n_test]

    # Reshape labels 1D to 2D array
    y_train = np.reshape(y_train, [-1,1])
    y_valid = np.reshape(y_valid, [-1,1]) 
    y_test = np.reshape(y_test, [-1,1])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def normalizer(X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_scl = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
    y_scl = MinMaxScaler(feature_range=(0, 1)).fit(y_train)

    X_train = X_scl.transform(X_train)
    X_valid = X_scl.transform(X_valid)
    X_test = X_scl.transform(X_test)

    y_train = y_scl.transform(y_train)
    y_valid = y_scl.transform(y_valid)
    y_test = y_scl.transform(y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, y_scl


def process(data, label, shift, dropzeros, log, split_valid, split_test):
    '''
    Wrapper function for all processing steps.
    '''

    # Preproc
    data = preprocessor(data, label, dropzeros, log)

    # Label
    data_features, data_labels = labeller(data, label, shift)

    # Split
    X_train, y_train, X_valid, y_valid, X_test, y_test = splitter(data_features, data_labels, split_valid, split_test)

    # Normalize
    X_train, y_train, X_valid, y_valid, X_test, y_test, y_scl = normalizer(X_train, y_train, X_valid, y_valid, X_test, y_test)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, y_scl