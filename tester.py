import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import libs
from libs import figurelib, modelib, NSRDBlib, processlib

model_selection = 'A'
model_seq = False

model_A_dir = './models/multiregressor.py/best/model.keras'
model_B_dir = './models/prometheus.py/model.keras'
model_C_dir = './models/prometheus_rnn.py/model.keras'

plot_label_x = 'Index'
plot_label_y = 'DHI [W/$m^2$]'

warmup_length = 84

features = [       # selected features must match those model trained with
    'ghi_clear',
    'dhi_clear',
    'dew_point',
    'precipitation',
    'dhi',
    'temp',
    'ghi',
    'dni_clear',
    'zenith_angle',
    'dni',
    'pressure',
    'humidity_rel',
    'albedo_sur',
    'wind_speed',
    'wind_dir',
    'cloud_type_0.0',
    'cloud_type_1.0',
    'cloud_type_2.0',
    'cloud_type_3.0',
    'cloud_type_4.0',
    'cloud_type_6.0',
    'cloud_type_7.0',
    'cloud_type_8.0',
    'cloud_type_9.0',
    'cloud_type_10.0'
]
features_label = 'dhi'
features_label_shift = 12       # hours
features_dropzeros = True       # whether to drop all rows in data for which label is 0
features_log = False

valid_split = 0.2
test_split = 0.2


file_name = os.path.basename(__file__)


def test_model(model, x, y):
    y_pred = model.predict(x=x, batch_size=None, verbose=0, steps=None)

    y = y_scl.inverse_transform(y)
    y_pred = y_scl.inverse_transform(y_pred)

    # if (features_log):        # must also check for zeros before exp all!!
    #     y_test = np.exp(y)
    #     y_pred = np.exp(y_pred)

    fig = figurelib.plot_pred(lbl=y, out=y_pred, xlbl=plot_label_x, ylbl=plot_label_y)

    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    figurelib.save_fig(fig=fig, name='output_plot.png', date=plot_date, id=file_name)

    plt.show()


def test_model_seq(model):      # WIP
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
    print('Testing model {}...'.format(model_selection))

    if model_selection == 'A':
        model = keras.models.load_model(model_A_dir)
    elif model_selection == 'B':
        model = keras.models.load_model(model_B_dir)
    elif model_selection == 'C':
        model = keras.models.load_model(model_C_dir)
    else:
        raise ValueError('Invalid model selection: '+model_selection)

    data = NSRDBlib.get_data(features)
    _, _, _, _, X_test, y_test, y_scl = processlib.process(
        data=data,
        label=features_label,
        shift=features_label_shift,
        dropzeros=features_dropzeros,
        log=features_log,
        split_valid=valid_split,
        split_test=test_split)
    
    if not model_seq:
        test_model(model=model, x=X_test, y=y_test)

    elif model_seq:
        test_model_seq(model)

    

