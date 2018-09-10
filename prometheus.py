import datetime as dt
import distutils
from distutils.util import strtobool
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
features_label_shift = 24       # hourly resolution
split_test = 0.2       # first
split_val = 0.25       # second

model_type = 'MLP'      # MLP/RNN/LSTM/GRU
learn_rate = 0.000033083129600225145
n_layers = 3
n_nodes = 100
act = 'relu'
dropout = 0.2

n_epochs = 100
batch_size = 256
sequence_length = 24       # hours in week
#endregion


#region Functions
def train_model(learn_rate=learn_rate, n_layers=n_layers, n_nodes=n_nodes, act=act, dropout=dropout):
    global X_train, y_train, X_test, y_test
    try:        # prevents data from being reprocessed every call
        X_train
    except NameError:
        data = NSRDBlib.get_data(features)      # get data
        X_train, y_train, X_test, y_test = processlib.process(data, features_label, features_label_shift, split_test, model_type, sequence_length)

    print('##################################')
    print('Model Hyperparameters:')
    print('Architecture: ', model_type)
    print('learning rate: {}'.format(learn_rate))
    print('layers:', n_layers)
    print('nodes:', n_nodes)
    print('activation:', act)
    print('dropout:', dropout)

    # Model instantiation
    if model_type == 'MLP':
        model = modelib.create_model_dense(learn_rate, n_layers, n_nodes, act, X_train.shape[1], dropout)
    elif model_type == 'RNN':
        model = modelib.create_model_RNN(learn_rate, n_layers, n_nodes, act, X_train.shape[2], X_train.shape[1])
    elif model_type == 'LSTM':
        model = modelib.create_model_LSTM(learn_rate, n_layers, n_nodes, act)
    elif model_type == 'GRU':
        model = modelib.create_model_GRU(learn_rate, n_layers, n_nodes, act)
    else:
        raise ValueError('Invalid model type {}'.format(model_type))


    global call_count       # keep track of hyperparam search calls
    try:
        call_count
    except NameError:
        call_count = 0

    # Callback logging
    log_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = './logs/{0}({1})_{2}_rate({3})_layers({4})_nodes({5})_drop({6})/'.format(log_date,os.path.basename(__file__),model_type,learn_rate,n_layers,n_nodes,dropout)
    callback_log = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=5,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False
    )

    # Early stopping
    # keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0,
    #     patience=2,
    #     verbose=1,
    #     mode='min',
    #     baseline=None)

    # Run model
    time_start = dt.datetime.now()
    history = model.fit(        # train model
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
        callbacks=[callback_log],
        validation_split=split_val,
        shuffle=False
    )
    time_end = dt.datetime.now()
    time_elapsed = time_end - time_start
    call_count += 1
    loss = history.history['val_loss'][-1]

    print('Model training completed!')
    print("Validation loss: {}".format(loss))
    print('Elapsed time: {}'.format(time_elapsed))
    if call_count >= 2:
        from epimetheus import params_search_calls
        print('Search {0}/{1}'.format(call_count,params_search_calls))

    return model, loss, log_dir


def test_model(model):
    print('Testing model...')
    result = model.evaluate(x=X_test, y=y_test)
    for name, value in zip(model.metrics_names, result):
        print(name, value)
    
    print('Plotting output...')
    output_train = model.predict(x=X_train, verbose=1)
    output_validate = model.predict(x=X_train, verbose=1)
    output_test = model.predict(x=X_test, verbose=1)

    data_comp_train = processlib.unprocess(output_train, y_train, model_type)
    data_comp_validate = processlib.unprocess(output_validate, y_train, model_type)
    data_comp_test = processlib.unprocess(output_test, y_test, model_type)

    matplotlib.style.use('classic')
    fig = plt.figure('{0} Output: {1}'.format(model_type,features_label))

    # ax1 = fig.add_subplot(3,1,1)
    # ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(1,1,1)
    
    # ax1.set_title('Train')
    # ax2.set_title('Validate')
    ax3.set_title('Test')

    #data_comp_validate = data_comp_validate[len(data_comp_validate) - len(data_comp_validate)*int(split_val):]

    # data_comp_train.plot(kind='line', ax=ax1)
    # data_comp_validate.plot(kind='line', ax=ax2)
    data_comp_test.plot(kind='line', ax=ax3)
    plt.show()


def save_model(model):
    if not os.path.exists('./models/'+os.path.basename(__file__)):     # model.save_weights() does not explicitly create dir
        os.makedirs('./models/'+os.path.basename(__file__))

    model.save('./models/'+os.path.basename(__file__)+'/model.keras')
    print('Model sucessfully saved!')
#endregion


if __name__ == '__main__':
    print('Commencing Prometheus model generation...')

    model, _, _ = train_model()

    test_model(model)

    save_request = input('Save model? [y/n]: ')
    if strtobool(save_request):
        save_model(model)

    print('Debug:\n$tensorboard --logdir=logs')