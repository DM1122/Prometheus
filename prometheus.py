import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import libs
from libs import figurelib, modelib, NNlib, NSRDBlib, processlib

np.random.seed(123)
tf.set_random_seed(123)


#region Hyperparams
model_type = 'MLP'      # MLP
n_epochs = 100
#n_epoch_steps = 64
learn_rate = 0.001
n_layers = 1
n_nodes = 16
activation = 'relu'
dropout_rate = 0.0
batch_size = 256

features = [
    'ghi_clear',
    'dhi_clear',
    'dew_point',
    'precipitation',
    'dhi',
    'temp',
    # 'ghi',
    # 'dni_clear',
    'zenith_angle',
    # 'dni',
    # 'pressure',
    'humidity_rel',
    'albedo_sur',
    # 'wind_speed',
    # 'wind_dir',
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
#endregion

file_name = os.path.basename(__file__)


def train_model(learn_rate=learn_rate, n_layers=n_layers, n_nodes=n_nodes, act=activation, dropout_rate=dropout_rate, batch_size=batch_size):
    '''
    Model generator and trainer function. Can be called from external hyperparam optimizer.
    '''

    global call_count       # keep track of hyperparam search calls
    try:
        call_count
    except NameError:
        call_count = 0

    #region Data handling
    global X_train, y_train, X_valid, y_valid, X_test, y_test, y_scl       # prevents data from being reprocessed every call
    try:
        X_train
    except NameError:
        data = NSRDBlib.get_data(features)
        X_train, y_train, X_valid, y_valid, X_test, y_test, y_scl = processlib.process(
            data=data,
            label=features_label,
            shift=features_label_shift,
            dropzeros=features_dropzeros,
            log=features_log,
            split_valid=valid_split,
            split_test=test_split)
    #endregion

    #region Model instantiation
    if model_type == 'MLP':
        model, opt = modelib.create_model_MLP(rate=learn_rate, layers=n_layers, nodes=n_nodes, act=activation, droprate=dropout_rate, inputs=X_train.shape[1], outputs=y_train.shape[1])
    else:
        raise ValueError('Invalid model type: {}'.format(model_type))
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    log_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log = './logs/'+file_name+'/{0}_{1}_rate({2})_act({3})_layers({4})_nodes({5})_drop({6})_batch({7})/'.format(
        log_date,
        model_type,
        learn_rate,
        act,
        n_layers,
        n_nodes,
        dropout_rate,
        batch_size)
    callbacks = NNlib.callbacks(log=log, id=file_name)
    #endregion

    #region Training
    print('=================================================')
    print('')
    print('Model Hyperparameters:')
    print('architecture: ', model_type)
    print('learning rate: ', learn_rate)
    print('layers: ', n_layers)
    print('nodes: ', n_nodes)
    print('activation: ', activation)
    print('dropout: ', dropout_rate)
    print('batch size: ', batch_size)

    # batchgen = modelib.batch_generator(x=X_train, y=y_train, batch_size=batch_size)      # create batch generator (X_batch, y_batch = next(batchgen))

    time_start = dt.datetime.now()

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_split=0.0,
        validation_data=(X_valid, y_valid),
        shuffle=False,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None)

    # model.fit_generator(
    #     generator=batchgen,
    #     steps_per_epoch=n_epoch_steps,
    #     epochs=n_epochs,
    #     verbose=1,
    #     callbacks=callbacks,
    #     validation_data=(X_valid, y_valid),
    #     validation_steps=None,
    #     class_weight=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False,
    #     shuffle=False,
    #     initial_epoch=0)

    time_end = dt.datetime.now()
    time_elapsed = time_end - time_start
    call_count += 1

    print('Model training completed!')
    #endregion
    
    #region Validation
    model.load_weights('./models/'+file_name+'/model.keras')        # restore best weights
    
    result = model.evaluate(x=X_valid, y=y_valid, batch_size=None, verbose=0, sample_weight=None, steps=None)       # validate

    for metric, val in zip(model.metrics_names, result):
        if metric == 'loss':
            loss = val       # get validation loss from metrics

    print('Validation loss: ', loss)
    print('Elapsed time: ', time_elapsed)
    if call_count >= 2:
        from epimetheus import params_search_calls
        print('Search {0}/{1}'.format(call_count, params_search_calls))
    #endregion

    return model, loss, log
    

if __name__ == '__main__':
    print('Commencing Prometheus model generation...')

    train_model()

    print('Debug:\n$tensorboard --logdir=logs/'+file_name)