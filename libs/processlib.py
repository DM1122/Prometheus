import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split as TrainTestSplit
from sklearn.preprocessing import MinMaxScaler

#region Functions
def labeller(data, label, shift):
    data['Label'] = data[label].shift(-shift)       # create label col
    data.dropna(inplace=True)       # drop gap created by shift length

    data_features = data.drop(columns='Label')      # create features df from all but labels col
    data_labels = data['Label']     # create labels df from labels col

    return data_features, data_labels


def splitter(data_features, data_labels, split):
    X_train, X_test, y_train, y_test = TrainTestSplit(data_features, data_labels, shuffle=False, test_size=split)

    # convert all datasets to np arrays for compatability
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # reshape labels 1D to 2D array
    y_train = np.reshape(y_train, [-1,1])     
    y_test = np.reshape(y_test, [-1,1])

    return X_train, y_train, X_test, y_test


def normalizer(X_train, y_train, X_test, y_test):
    X_scl = MinMaxScaler(feature_range=(0, 1)).fit(X_train)

    X_train = X_scl.transform(X_train)
    X_test = X_scl.transform(X_test)

    return X_train, y_train, X_test, y_test


def reshaper(data_raw, timesteps):
    crop = data_raw.shape[0]-data_raw.shape[0]%timesteps        # modulus
    data_raw = data_raw[:crop]      # crop to length divisble by timesteps

    # data = []
    # for i in range(data_raw.shape[0] // timesteps):     # int divisor // or subtraction -
    #     data.append(data_raw[i*timesteps:i*timesteps+timesteps])
    # data = np.array(data)
    # data = np.reshape(data, (data.shape[0], data.shape[1], data_raw.shape[1]))      # samples, timesteps, features (unneccesary?)

    data = []
    for i in range(data_raw.shape[0] - timesteps):     # int divisor // or subtraction -
        data.append(data_raw[i:i+timesteps])
    data = np.array(data)
    
    return data


def unshaper(data_raw):
    data_raw = np.reshape(data_raw, (data_raw.shape[0], data_raw.shape[1]))       # samples, features

    data = []
    data.append(data_raw[0:1])      # append 
    data = np.array(data)
    data = np.reshape(data, (data.shape[2],1))

    for i in range(data_raw.shape[0] - 1):
        block = []
        block.append(data_raw[i+1:i+2, data_raw.shape[1]-1:])
        block = np.array(block)
        block = np.reshape(block, (block.shape[2],1))

        data = np.concatenate((data, block), axis=0)

    return data


def process(data, label, shift, split, model, timesteps):

    # Label
    data_features, data_labels = labeller(data, label, shift)

    # Split
    X_train, y_train, X_test, y_test = splitter(data_features, data_labels, split)

    # Normalize
    X_train, y_train, X_test, y_test = normalizer(X_train, y_train, X_test, y_test)

    # Reshape RNNs
    if model == 'RNN' or model == 'LSTM' or model == 'GRU':
        X_train = reshaper(X_train, timesteps)
        y_train = reshaper(y_train, timesteps)
        X_test = reshaper(X_test, timesteps)
        y_test = reshaper(y_test, timesteps)
    
    return X_train, y_train, X_test, y_test


def unprocess(output_raw, y_raw, model):

    # Unshape RNNs
    if model == 'RNN' or model == 'LSTM' or model == 'GRU':
        output_raw = unshaper(output_raw)
        y_raw = unshaper(y_raw)
    
    # Convert to df
    output = pd.DataFrame(output_raw, columns=['Output'])
    y = pd.DataFrame(y_raw, columns=['Label'])

    # Concat dfs
    data_comp = pd.concat([output, y], axis=1, sort=False)
    data_comp.index.names = ['Index']

    return data_comp

#endregion