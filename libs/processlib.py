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


def reshape(data_raw, look_back):
    crop = data_raw.shape[0]-data_raw.shape[0]%look_back        # modulus
    data_raw = data_raw[:crop]      # crop to length divisble by look_back

    data = []
    for i in range(data_raw.shape[0] // look_back):     # int divisor // or subtraction -
        #data.append(data_raw[i:(i+look_back)])
        data.append(data_raw[i*look_back:i*look_back+look_back])
    data = np.array(data)

    data = np.reshape(data, (data.shape[0], data.shape[1], data_raw.shape[1]))      # samples, timesteps, features (unneccesary?)
    
    return data


def unshape(data_raw, look_back):
    data = np.reshape(data_raw, (data_raw.shape[0]*data_raw.shape[1], data_raw.shape[2]))       # samples, features

    return data


def unprocess(out_raw, y_raw):    
    out = pd.DataFrame(scl.inverse_transform(out_raw), columns=['Output'])
    y = pd.DataFrame(scl.inverse_transform(y_raw), columns=['Truth'])

    data_forecast = pd.concat([out, y], axis=1, sort=False)
    data_forecast.index.names = ['Index']

    return data_forecast
#endregion