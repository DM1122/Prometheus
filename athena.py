import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import modelib
import tensorflow as tf
from tensorflow import keras
import skopt
from skopt.utils import use_named_args
import shutil
import os
import NSRDBlib
import processlib

np.random.seed(123)
tf.set_random_seed(123)

#region Metaparams
features = [       # Temperature/Clearsky DHI/Clearsky DNI/Clearsky GHI/Dew Point/DHI/DNI/GHI/Relative Humidity/Solar Zenith Angle/Surface Albedo/Pressure/Precipitable Water/Wind Direction/Wind Speed/Cloud Type_(0-10).0
    'Temperature',
    'Clearsky DHI',
    'Clearsky DNI',
    'Clearsky GHI',
    'Dew Point',
    'DHI',
    'DNI',
    'GHI',
    'Relative Humidity',
    'Solar Zenith Angle',
    'Surface Albedo',
    'Pressure',
    'Precipitable Water',
    'Wind Direction',
    'Wind Speed',
    'Cloud Type_0.0',
    'Cloud Type_1.0',
    'Cloud Type_2.0',
    'Cloud Type_3.0',
    'Cloud Type_4.0',
    'Cloud Type_6.0',
    'Cloud Type_7.0',
    'Cloud Type_8.0',
    'Cloud Type_9.0',
    'Cloud Type_10.0'
]
features_label = 'DHI'
features_label_shift = 24       # hourly resolution

split_test = 0.2       # first
split_val = 0.25       # second

n_epochs = 100
batch_size = 128
params_search_calls = 500       # must be >=11 (risk 'The objective has been evaluated at this point before' w/ values >100)

nodes_architecture = 'MLP'      # LN/MLP/RNN/LSTM/GRU
#endregion

#region Hyperparams
learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
layers_space = skopt.space.Integer(low=1, high=8, name='n_layers')
nodes_space = skopt.space.Integer(low=4, high=512, name='n_nodes')
act_space = skopt.space.Categorical(categories=['relu'], name='act')

params = [learn_rate_space, layers_space, nodes_space, act_space]
params_init = [3e-5, 1, 16, 'relu']
#endregion

#region Functions
@use_named_args(dimensions=params)      # allows params to be passed as list
def fitness(learn_rate, n_layers, n_nodes, act):
    log_dir = './logs/lr_{0:0e}_layers_{1}_nodes_{2}_{3}/'.format(learn_rate, n_layers, n_nodes, act)       # update log dir

    if nodes_architecture == 'LN':
        model = modelib.create_model_linear(learn_rate, len(X_test[0]))
    elif nodes_architecture == 'MLP':
        model = modelib.create_model_dense(learn_rate, n_layers, n_nodes, act, len(X_test[0]))
    elif nodes_architecture == 'RNN':
        model = modelib.create_model_rnn(learn_rate, n_layers, n_nodes, act, len(X_test[0]))
    else:
        raise ValueError('Invalid model type {}'.format(nodes_architecture))

    print('##################################')
    print('Training new model:')
    print('learning rate: {0:.1e}'.format(learn_rate))
    print('layers:', n_layers)
    print('nodes:', n_nodes)
    print('activation:', act)

    callback_log = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,       # not working currently
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False
    )

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

    loss = history.history['val_loss'][-1]        # get most recent accuracy from history
    print("Validation Loss: {0:.2%}".format(loss))
    update_best_model(model, loss, log_dir)
    
    del model
    keras.backend.clear_session()

    return loss

def update_best_model(model, loss, log_dir):
    global loss_best

    # will set best loss to current loss on first run
    try:
        loss_best
    except NameError:
        loss_best = loss

    # determine best model
    if loss < loss_best:
        if not os.path.exists('./models/'):     # model.save() does not explicitly create dir
            os.makedirs('./models/')

        print('New best!')
        loss_best = loss      # update best accuracy
        model.save('./models/model.keras')     # save best model to disk

        # delete all other model logs
        for log in os.listdir('./logs/'):
            if ('./logs/'+log+'/') != log_dir:
                shutil.rmtree('./logs/'+log+'/')
    else:
        shutil.rmtree(log_dir)      # delete current log

def test_model():       # does not work in current version of keras
    print('Testing model...')
    model = keras.models.load_model('./models/model.keras')
    result = model.evaluate(x=X_test, y=y_test)
    for name, value in zip(model.metrics_names, result):
        print(name, value)

def plot_results(search):
    matplotlib.style.use('classic')
    fig1 = skopt.plots.plot_convergence(search)
    fig2, ax = skopt.plots.plot_histogram(result=search, dimension_name='n_layers')
    fig3, ax = skopt.plots.plot_objective(result=search, dimension_names=['learn_rate','n_layers','n_nodes'])
    fig4, ax = skopt.plots.plot_evaluations(result=search, dimension_names=['learn_rate','n_layers','n_nodes'])
    plt.show()

#endregion

#region Main
data = NSRDBlib.get_data(features)      # get data
data_features, data_labels = processlib.label(data, features_label, features_label_shift)       # create labels
X_train_raw, y_train_raw, X_test_raw, y_test_raw = processlib.split(data_features, data_labels, split_test)        # split data into train/test
X_train, y_train, X_test, y_test = processlib.normalize(X_train_raw,y_train_raw, X_test_raw, y_test_raw)        # normalize datasets

print('Commencing Athena hyperparameter optimization...')
params_search = skopt.gp_minimize(      # bayesian optimization
    func=fitness,
    dimensions=params,
    acq_func='EI',
    n_calls=params_search_calls,
    x0=params_init
)

print('Hyperparameter optimization completed!')
print('Results: ', params_search.space.point_to_dict(params_search.x))
print('Fitness: ', params_search.fun)
plot_results(params_search)

print('Debug:\n$tensorboard --logdir=logs')
#endregion