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

def unprocess(out_raw, y_raw):       # untested
    
    out = pd.DataFrame(scl.inverse_transform(out_raw), columns=['Output'])
    y = pd.DataFrame(scl.inverse_transform(y_raw), columns=['Truth'])

    data_forecast = pd.concat([out, y], axis=1, sort=False)
    data_forecast.index.names = ['Index']

    print(data_forecast)
    return data_forecast
#endregion

# def unprocess_data(out_train, out_test, y_train, y_test, label, scl, index):

#     # index split
#     index_train = index.loc[:len(out_train)-1,:]
#     index_test = index.loc[len(out_train):,:]
#     index_test.reset_index(drop=True, inplace=True)     # reset sliced index to start at 0

#     # output unproccessing
#     out_train_real = pd.DataFrame(scl.inverse_transform(out_train), columns=['Prediction'])
#     out_test_real = pd.DataFrame(scl.inverse_transform(out_test), columns=['Prediction'])
    
#     # labels unprocessing
#     y_train_real = pd.DataFrame(scl.inverse_transform(y_train), columns=[label])
#     y_test_real = pd.DataFrame(scl.inverse_transform(y_test), columns=[label])

#     # forecast comp
#     forecast_train = pd.concat([index_train, y_train_real, out_train_real], axis=1, sort=True)
#     forecast_train.set_index('index', drop=True, inplace=True)
#     forecast_train.index.names = ['Date']

#     forecast_test = pd.concat([index_test, y_test_real, out_test_real], axis=1, sort=True)
#     forecast_test.set_index('index', drop=True, inplace=True)
#     forecast_test.index.names = ['Date']

#     return forecast_train, forecast_test