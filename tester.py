import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import libs
from libs import figurelib, modelib, NSRDBlib, processlib


def test_model(model):
    print('Testing model...')

    #region Testing
    y_pred = model.predict(x=X_test, batch_size=None, verbose=0, steps=None)        # predict using test data
 
    global y_test       # unscale datasets
    y_test = y_scl.inverse_transform(y_test)
    y_pred = y_scl.inverse_transform(y_pred)

    if (features_log):
        y_test = np.exp(y_test)
        y_pred = np.exp(y_pred)
    #endregion

    #region Plots
    fig = figurelib.plot_pred(lbl=y_test, out=y_pred, xlbl='Index', ylbl='DHI [W/$m^2$]')

    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    figurelib.save_fig(fig=fig, name='output_plot.png', date=plot_date, id=file_name)

    plt.show()
    #endegion


def test_model_seq(model):
    print('Testing sequential model...')

    #region Testing
    y_pred = model.predict(x=np.expand_dims(X_test, axis=0), batch_size=None, verbose=0, steps=None)        # predict using test data
    y_pred = y_pred.reshape(y_pred.shape[1],1)
 
    global y_test       # unscale datasets
    y_test = y_scl.inverse_transform(y_test)
    y_pred = y_scl.inverse_transform(y_pred)

    if (features_log):
        y_test = np.exp(y_test)
        y_pred = np.exp(y_pred)
    #endregion

    #region Plots
    fig = figurelib.plot_pred_warmup(lbl=y_test, out=y_pred, warmup=warmup_length, xlbl='Index', ylbl='DHI [W/$m^2$]')

    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    figurelib.save_fig(fig=fig, name='output_plot.png', date=plot_date, id=file_name)

    plt.show()
    #endregion


if __name__ == '__main__':
    print('Commencing Prometheus model generation...')

    model, _, _ = train_model()

    print('Debug:\n$tensorboard --logdir=logs/'+file_name)