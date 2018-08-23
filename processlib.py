import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split as TrainTestSplit
from sklearn.preprocessing import MinMaxScaler

#region Functions
def label(data, label, shift):
    data['Label'] = data[label].shift(-shift)       # create label col
    data.dropna(inplace=True)       # drop gap created by shift length

    data_features = data.drop(columns='Label')      # create features df from all but labels col
    data_labels = data['Label']     # create labels df from labels col

    return data_features, data_labels

def split(data_features, data_labels, split):
    X_tr, X_te, y_tr, y_te = TrainTestSplit(data_features, data_labels, shuffle=False, test_size=split)

    # convert all datasets to np arrays for compatability
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)
    X_te = np.array(X_te)
    y_te = np.array(y_te)

    # reshape labels 1D to 2D array
    y_tr = np.reshape(y_tr, [-1,1])     
    y_te = np.reshape(y_te, [-1,1])

    return X_tr, y_tr, X_te, y_te

def normalize(X_tr, y_tr, X_te, y_te):
    X_scl = MinMaxScaler(feature_range=(0, 1)).fit(X_tr)
    y_scl = MinMaxScaler(feature_range=(0, 1)).fit(y_tr)

    X_tr = X_scl.transform(X_tr)
    y_tr = y_scl.transform(y_tr)
    X_te = X_scl.transform(X_te)    
    y_te = y_scl.transform(y_te)

    global scl
    scl = y_scl

    return X_tr, y_tr, X_te, y_te

def unormalize(data):       # untested
    print(scl)
    data = scl.inverse_transform(data)

    return data
#endregion