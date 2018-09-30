import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import libs
from libs import figurelib, modelib, NSRDBlib, processlib

np.random.seed(123)
tf.set_random_seed(123)


#region Hyperparams
model_type = 'MLP'      # MLP
n_epochs = 100
n_epoch_steps = 64
learn_rate = 0.001
n_layers = 1
n_nodes = 16
activation = 'relu'
dropout_rate = 0.0
batch_size = 256

features = [       # temp/dhi_clear/dni_clear/ghi_clear/dew_point/dhi/dni/ghi/humidity_rel/zenith_angle/albedo_sur/pressure/precipitation/wind_dir/win_speed/cloud_type_(0-10).0 (exclude 5)
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
features_dropzeros = False       # whether to drop all rows in data for which label is 0
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
    callbacks = modelib.callbacks(log=log, id=file_name)
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


# def train_model(learn_rate=learn_rate, n_layers=n_layers, n_nodes=n_nodes, act=act, dropout=dropout):
#     global X_train, y_train, X_test, y_test
#     try:        # prevents data from being reprocessed every call
#         X_train
#     except NameError:
#         data = NSRDBlib.get_data(features)      # get data
#         X_train, y_train, X_test, y_test = processlib.process(data, features_label, features_label_shift, features_drop_zeros, split_test, model_type, sequence_length)

#     print('=================================================')
#     print('')
#     print('Model Hyperparameters:')
#     print('architecture: ', model_type)
#     print('learning rate: {}'.format(learn_rate))
#     print('layers:', n_layers)
#     print('nodes:', n_nodes)
#     print('activation:', act)
#     print('dropout:', dropout)

#     # Model instantiation
#     if model_type == 'MLP':
#         model = modelib.create_model_dense(learn_rate, n_layers, n_nodes, act, X_train.shape[1], dropout)
#     elif model_type == 'RNN':
#         model = modelib.create_model_RNN(learn_rate, n_layers, n_nodes, act, X_train.shape[2], X_train.shape[1])
#     elif model_type == 'LSTM':
#         model = modelib.create_model_LSTM(learn_rate, n_layers, n_nodes, act)
#     elif model_type == 'GRU':
#         model = modelib.create_model_GRU(learn_rate, n_layers, n_nodes, act)
#     else:
#         raise ValueError('Invalid model type {}'.format(model_type))


#     global call_count       # keep track of hyperparam search calls
#     try:
#         call_count
#     except NameError:
#         call_count = 0

#     # Callback logging
#     log_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
#     log_dir = './logs/'+file_name+'/'
#     log_file = log_dir+'{0}_{1}_rate({2})_layers({3})_nodes({4})_drop({5})/'.format(log_date,model_type,learn_rate,n_layers,n_nodes,dropout)
#     callback_log = keras.callbacks.TensorBoard(
#         log_dir=log_file,
#         histogram_freq=5,
#         batch_size=32,
#         write_graph=True,
#         write_grads=True,
#         write_images=False
#     )


#     # Run model
#     time_start = dt.datetime.now()
#     history = model.fit(        # train model
#         x=X_train,
#         y=y_train,
#         batch_size=batch_size,
#         epochs=n_epochs,
#         verbose=1,
#         callbacks=[callback_log],
#         validation_split=split_val,
#         shuffle=False
#     )
#     time_end = dt.datetime.now()
#     time_elapsed = time_end - time_start
#     call_count += 1
#     loss = history.history['val_loss'][-1]

#     print('Model training completed!')
#     print("Validation loss: {}".format(loss))
#     print('Elapsed time: {}'.format(time_elapsed))
#     if call_count >= 2:
#         from epimetheus import params_search_calls
#         print('Search {0}/{1}'.format(call_count,params_search_calls))

#     return model, loss, log_file


# def test_model(model):
#     print('Testing model...')
#     result = model.evaluate(x=X_test, y=y_test)
#     for metric, value in zip(model.metrics_names, result):
#         print('{0}: {1}'.format(metric, value))
    
#     print('Plotting output...')
#     output_train = model.predict(x=X_train, verbose=1)
#     output_test = model.predict(x=X_test, verbose=1)

#     data_comp_train_unsliced = processlib.unprocess(output_train, y_train, model_type)
#     print(data_comp_train_unsliced.shape)
#     print(len(data_comp_train_unsliced))

#     data_comp_train = data_comp_train_unsliced[:int(len(data_comp_train_unsliced) - len(data_comp_train_unsliced)*(split_val))]
#     print(data_comp_train.shape)

#     data_comp_validate = data_comp_train_unsliced[int(len(data_comp_train_unsliced) - len(data_comp_train_unsliced)*(split_val)):]
#     print(data_comp_validate.shape)

#     data_comp_test = processlib.unprocess(output_test, y_test, model_type)

#     matplotlib.style.use('classic')

#     # Generate training Figure
#     fig1 = plt.figure()
#     ax1 = fig1.add_subplot(1,1,1)
#     ax1.set_title('Training')
#     data_comp_train.plot(kind='line', ax=ax1)
#     data_comp_validate.plot(kind='line', ax=ax1)
#     fig1.tight_layout()

#     # Generate testing Figure
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot(1,1,1)
#     ax2.set_title('Testing')
#     data_comp_test.plot(kind='line', ax=ax2)
#     fig2.tight_layout()

#     # Save plot to disk
#     plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
#     plot_dir = './plots/'+file_name+'/'+plot_date+'/'
#     os.makedirs(plot_dir)
#     fig1.savefig(plot_dir+'training_plot.png')
#     fig2.savefig(plot_dir+'testing_plot.png')

#     plt.show()




if __name__ == '__main__':
    print('Commencing Prometheus model generation...')

    model, _, _ = train_model()

    print('Debug:\n$tensorboard --logdir=logs/'+file_name)