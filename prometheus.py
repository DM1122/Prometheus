import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import skopt
from skopt.utils import use_named_args
import shutil
import datalib as data
import os

np.random.seed(123)
tf.set_random_seed(123)

#region Data
ticker_prim = 'GOOGL'
tickers_sec = []

features_prim = ['Close', 'Open', 'High', 'Low', 'Volume', 'HML', 'PCT']
features_sec = ['Close', 'Volume', 'HML', 'PCT']

features_label = 'Close'
features_label_shift = 7     # forecast distance for labels

features_cutoff = 1095
features_split = 30     # data points to be used in test portion of train/test split
#endregion

n_epochs = 100
batch_size = 128
params_search_calls = 11        # must be >=11

#region Hyperparameter search spaces
learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
layers_space = skopt.space.Integer(low=1, high=5, name='n_layers')
nodes_space = skopt.space.Integer(low=4, high=512, name='n_nodes')
act_space = skopt.space.Categorical(categories=['relu'], name='act')

params = [learn_rate_space, layers_space, nodes_space, act_space]
params_init = [3e-5, 1, 16, 'relu']
#endregion

#region Functions
def create_model(learn_rate, n_layers, n_nodes, act):
    model = keras.Sequential()
    
    model.add(keras.layers.InputLayer(input_shape=(len(X_train[0]),)))

    for i in range(n_layers):
        model.add(keras.layers.Dense(units=n_nodes, activation=act, name='layer_dense_{}'.format(i+1)))
    
    model.add(keras.layers.Dense(units=1, activation='linear', name='layer_output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

@use_named_args(dimensions=params)      # allows params to be passed as list
def fitness(learn_rate, n_layers, n_nodes, act):

    log_dir = './logs/lr_{0:0e}_layers_{1}_nodes_{2}_{3}/'.format(learn_rate, n_layers, n_nodes, act)

    model = create_model(learn_rate, n_layers, n_nodes, act)

    print('##################################')
    print('Training new model:')
    print('learning rate: {0:.1e}'.format(learn_rate))
    print('layers:', n_layers)
    print('nodes:', n_nodes)
    print('activation:', act)

    callback_log = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
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
        validation_split=0.1,
        shuffle=False
    )

    loss = history.history['val_loss'][-1]        # get most recent accuracy from history

    print("Validation Loss: {0:.2%}".format(loss))

    global loss_best        # handler for first model run
    try:
        loss_best
    except NameError:
        loss_best = loss

    if loss < loss_best:
        if not os.path.exists('./models/'):     # model.save() does not explicitly create dir
            os.makedirs('./models/')

        print('New best!')
        loss_best = loss      # update best accuracy
        model.save('./models/model.keras')     # save best model to disk

        for log in os.listdir('./logs/'):
            if ('./logs/'+log+'/') != log_dir:
                shutil.rmtree('./logs/'+log+'/')
    else:
        shutil.rmtree(log_dir)      # delete current log
    
    del model       # delete model with these hyper-parameters from memory.
    keras.backend.clear_session()       # clear the Keras session, otherwise adds new models to preexisting graph

    return loss

def test_model():       # does not work in newest version of keras
    print('Testing model:')
    model = keras.models.load_model('./models/model.keras')
    result = model.evaluate(x=X_test, y=y_test)
    for name, value in zip(model.metrics_names, result):
        print(name, value)
#endregion

#region Main
X_train, X_test, y_train, y_test, data_scalar, data_index = data.process_data(
    ticker_prim, tickers_sec,
    features_prim, features_sec,
    features_label, features_label_shift,
    features_cutoff, features_split)

#loss_best = 100     # initial value of best percentage (to be reduced)

params_search = skopt.gp_minimize(      # use bayesian optimization to approximate best hyperparams
    func=fitness,
    dimensions=params,
    acq_func='EI',
    n_calls=params_search_calls,
    x0=params_init
)

print('Hyperparameter optimization completed!')
print('Results: ', params_search.space.point_to_dict(params_search.x))
print('Fitness: ', params_search.fun)
matplotlib.style.use('classic')
fig1 = skopt.plots.plot_convergence(params_search)
fig2, ax = skopt.plots.plot_histogram(result=params_search, dimension_name='n_layers')
fig3, ax = skopt.plots.plot_objective(result=params_search, dimension_names=['learn_rate','n_layers','n_nodes'])
fig4, ax = skopt.plots.plot_evaluations(result=params_search, dimension_names=['learn_rate','n_layers','n_nodes'])
plt.show()

print('Debug:\n$tensorboard --logdir=logs')
#endregion