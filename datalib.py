import stocklib as stock

import sklearn
from sklearn.model_selection import train_test_split as TrainTestSplit
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


def process_data(ticker_prim, tickers_sec, features_prim, features_sec, label, shift, cutoff, split):
    
    # primary ticker processing
    data_prim_raw = stock.get_data(ticker_prim)     # get primary stock df  
    data_prim = pd.DataFrame()     # create empty df to be populated w/ filtered features
    for feature in features_prim:   # feature filter
        data_prim = pd.concat([data_prim, data_prim_raw[feature]], axis=1, sort=True)

    # secondary ticker processing
    data_sec = pd.DataFrame()
    if len(tickers_sec) != 0:     # only run if secondary tckrs exist        
        for ticker in tickers_sec:
            data_sec_raw = stock.get_data(ticker)
            if len(features_sec) != 0:
                for feature in features_sec:    # feature filter
                    data_sec_raw.rename({feature:feature+'({})'.format(ticker)}, axis=1, inplace=True)
                    data_sec = pd.concat([data_sec, data_sec_raw[feature+'({})'.format(ticker)]], axis=1, sort=True)
            else:
                raise ValueError('Secondary ticker(s) assigned but no feature(s) indicated')

    # concatetnation of dataframes
    data_comp = pd.concat([data_prim, data_sec], axis=1, sort=True)
    data_comp.dropna(axis=0, inplace=True)

    # label creation
    data_comp['Label'] = data_comp[label].shift(-shift)
    data_comp.dropna(inplace=True)
    

    # postproccessing
    data_comp.reset_index(drop=False, inplace=True)     # create column from date index
    if len(tickers_sec) == 0:   # bug renames Date column to index when concatenating
        data_comp.rename(columns={'Date':'index'}, inplace=True)
    data_comp = data_comp.loc[len(data_comp)-cutoff:,:]     # slice df by cutoff
    data_comp.reset_index(drop=True, inplace=True)  # reset sliced index to start at 0
    index = pd.DataFrame(data_comp['index'])    # save index for plotting
    index['index'] = pd.to_datetime(index['index'])     # convert str to datetime

    data_comp.drop(columns='index', inplace=True)
    data_features = np.array(data_comp.drop(columns='Label'))   # get all but label's column
    data_labels = np.array(data_comp['Label'])  # get nothing but label's column

    # tr/te split
    split_proportion = split / len(data_comp)   # calculate proportion of split value
    X_tr, X_te, y_tr, y_te = TrainTestSplit(data_features, data_labels, shuffle=False, test_size=split_proportion)

    # 1D to 2D array
    y_tr = np.reshape(y_tr, [-1,1])
    y_te = np.reshape(y_te, [-1,1])

    # normalization
    if label == 'PCT':  # normalize 'PCT' between (-1)-1
        X_scl = MinMaxScaler(feature_range=(-1, 1)).fit(X_tr)
        y_scl = MinMaxScaler(feature_range=(-1, 1)).fit(y_tr)
        X_tr = X_scl.transform(X_tr)
        X_te = X_scl.transform(X_te)
        y_tr = y_scl.transform(y_tr)
        y_te = y_scl.transform(y_te)
    else:   # normalize else between 0-1
        X_scl = MinMaxScaler(feature_range=(0, 1)).fit(X_tr)
        y_scl = MinMaxScaler(feature_range=(0, 1)).fit(y_tr)
        X_tr = X_scl.transform(X_tr)
        X_te = X_scl.transform(X_te)
        y_tr = y_scl.transform(y_tr)
        y_te = y_scl.transform(y_te)

    return X_tr, X_te, y_tr, y_te, y_scl, index

def unprocess_data(out_train, out_test, y_train, y_test, label, scl, index):

    # index split
    index_train = index.loc[:len(out_train)-1,:]
    index_test = index.loc[len(out_train):,:]
    index_test.reset_index(drop=True, inplace=True)     # reset sliced index to start at 0

    # output unproccessing
    out_train_real = pd.DataFrame(scl.inverse_transform(out_train), columns=['Prediction'])
    out_test_real = pd.DataFrame(scl.inverse_transform(out_test), columns=['Prediction'])
    
    # labels unprocessing
    y_train_real = pd.DataFrame(scl.inverse_transform(y_train), columns=[label])
    y_test_real = pd.DataFrame(scl.inverse_transform(y_test), columns=[label])

    # forecast comp
    forecast_train = pd.concat([index_train, y_train_real, out_train_real], axis=1, sort=True)
    forecast_train.set_index('index', drop=True, inplace=True)
    forecast_train.index.names = ['Date']

    forecast_test = pd.concat([index_test, y_test_real, out_test_real], axis=1, sort=True)
    forecast_test.set_index('index', drop=True, inplace=True)
    forecast_test.index.names = ['Date']

    return forecast_train, forecast_test