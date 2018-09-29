import datetime as dt
import libs
from libs import modelib, NSRDBlib, processlib
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

np.random.seed(123)
tf.set_random_seed(123)

#region Hyperparams
model_type = 'GRU'      # RNN/LSTM/GRU
n_epochs = 20
n_epoch_steps = 100
learn_rate = 0.001
n_layers = 1
n_nodes = 512
activation = 'tanh'
dropout_rate = 0.0
batch_size = 256
sequence_length = 672       # one month in hrs (assuming 12h day)

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
features_label_scale = True
features_dropzeros = True       # whether to drop all rows in data for which label is 0

valid_split = 0.2
test_split = 0.2
#endregion


def batch_generator(x, y, batch_size, timesteps):
    '''
    Generator function for creating random batches of data.
    Author: Hvass-Labs
    Modifier: DM1122
    '''

    # Infinite loop
    while True:
        # Allocate new empty array for batch of input signals
        x_batch = np.zeros(shape=(batch_size, timesteps, x.shape[1]))

        # Allocate new empty array for batch of output signals
        y_batch = np.zeros(shape=(batch_size, timesteps, y.shape[1]))

        # Fill batch arrays with random sequences of data
        for i in range(batch_size):
            # Get random start index
            idx = np.random.randint(x.shape[0] - timesteps)
            
            # Copy the sequences of data starting at idx
            x_batch[i] = x[idx:idx+timesteps]
            y_batch[i] = y[idx:idx+timesteps]
        
        yield (x_batch, y_batch)


def calculate_loss(y_true, y_pred):     # WIP
    '''
    Calculate the MSE between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    '''

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    # y_true_slice = y_true[:, sequence_length:, :]
    # y_pred_slice = y_pred[:, sequence_length:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    # print(y_true.shape)
    # print(y_pred.shape)

    # if (features_label_scale):      # unscale datasets (does not work)

    #     y_true = y_scl.inverse_transform(y_test)
    #     y_pred = y_scl.inverse_transform(y_pred)

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mse_mean = tf.reduce_mean(loss)

    return loss_mse_mean


def train_model(learn_rate=learn_rate, n_layers=n_layers, n_nodes=n_nodes, act=activation, dropout_rate=dropout_rate, batch_size=batch_size, sequence_length=sequence_length):
    '''
    Model generator and trainer function. Can be called from external hyperparam optimizer.
    '''
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
            labelscl=features_label_scale,
            split_valid=valid_split,
            split_test=test_split)
    #endregion

    #region Model instantiation
    if model_type == 'RNN':
        model, opt = modelib.create_model_RNN(rate=learn_rate, layers=n_layers, nodes=n_nodes, act=activation, droprate=dropout_rate, inputs=X_train.shape[1], outputs=y_train.shape[1])
    elif model_type == 'LSTM':
        model, opt = modelib.create_model_LSTM(rate=learn_rate, layers=n_layers, nodes=n_nodes, act=activation, droprate=dropout_rate, inputs=X_train.shape[1], outputs=y_train.shape[1])
    elif model_type == 'GRU':
        model, opt = modelib.create_model_GRU(rate=learn_rate, layers=n_layers, nodes=n_nodes, act=activation, droprate=dropout_rate, inputs=X_train.shape[1], outputs=y_train.shape[1])
    else:
        raise ValueError('Invalid model type: {}'.format(model_type))
    
    model.compile(optimizer=opt, loss=calculate_loss, metrics=['mae'])
    #endregion

    #region Callbacks
    if not os.path.exists('./logs/prometheus_rnn/'):
        os.makedirs('./logs/prometheus_rnn/')
    if not os.path.exists('./models/prometheus_rnn/'):
        os.makedirs('./models/prometheus_rnn/')

    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./models/prometheus_rnn/model.keras',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,    # might have to save weights only
        mode='auto',
        period=1)
    
    callback_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto',
        baseline=None)     # check for restore_best_weights in next release
    
    callback_NaN = keras.callbacks.TerminateOnNaN()

    log_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = './logs/prometheus_rnn/{0}_{1}_rate({2})_act({3})_layers({4})_nodes({5})_drop({6})_batch({7})_seq({8})/'.format(
        log_date,
        model_type,
        learn_rate,
        act,
        n_layers,
        n_nodes,
        dropout_rate,
        batch_size,
        sequence_length)

    callback_tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=5,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False)

    callbacks = [callback_checkpoint, callback_early_stopping, callback_NaN, callback_tensorboard]

    global call_count       # keep track of hyperparam search calls
    try:
        call_count
    except NameError:
        call_count = 0
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
    print('sequence length: ', sequence_length)

    batchgen = batch_generator(x=X_train, y=y_train, batch_size=batch_size, timesteps=sequence_length)      # create batch generator (X_batch, y_batch = next(batchgen))

    time_start = dt.datetime.now()

    model.fit_generator(        # train model
        generator=batchgen,
        steps_per_epoch=n_epoch_steps,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(np.expand_dims(X_valid, axis=0), np.expand_dims(y_valid, axis=0)),
        validation_steps=None,
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=False,
        initial_epoch=0)

    time_end = dt.datetime.now()
    time_elapsed = time_end - time_start
    call_count += 1

    print('Model training completed!')
    #endregion
    
    #region Validation
    try:
        model.load_weights('./models/prometheus_rnn/model.keras')     # restore weights from best checkpoint
    except Exception as error:
        print('Error trying to load checkpoint')
        print(error)
    
    result = model.evaluate(x=np.expand_dims(X_valid, axis=0), y=np.expand_dims(y_valid, axis=0), batch_size=None, verbose=0, sample_weight=None, steps=None)       # validate

    for metric, val in zip(model.metrics_names, result):
        if metric == 'loss':
            loss_valid = val       # get validation loss from metrics

    print('Validation loss: ', loss_valid)
    print('Elapsed time: ', time_elapsed)
    if call_count >= 2:
        from epimetheus_rnn import params_search_calls
        print('Search {0}/{1}'.format(call_count, params_search_calls))
    #endregion

    return model, loss_valid, log_dir


def test_model(model):
    print('Testing model...')
    y_pred = model.predict(x=np.expand_dims(X_test, axis=0), batch_size=None, verbose=0, steps=None)        # predict using test data
    y_pred = y_pred.reshape(y_pred.shape[1],1)

    #region Plotting
    matplotlib.style.use('classic')

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Index')
    ax.set_ylabel('DHI [W/$m^2$]')
    

    # Plot and compare two signals
    if (features_label_scale):      # unscale datasets
        global y_test
        y_test = y_scl.inverse_transform(y_test)
        y_pred = y_scl.inverse_transform(y_pred)

    plt.plot(y_test, label='Label')
    plt.plot(y_pred, label='Output')
    plt.axvspan(0, sequence_length, facecolor='black', alpha=0.15)

    fig.tight_layout()
    plt.legend()

    save_plot(fig)
    plt.show()
    #endregion


if __name__ == '__main__':
    print('Commencing Prometheus model generation...')

    model, _, _ = train_model()

    test_model(model)

    print('Debug:\n$tensorboard --logdir=logs/prometheus_rnn')