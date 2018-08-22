import tensorflow as tf
from tensorflow import keras

import skopt

#region Defining hyper-parameter search space
learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
layers_space = skopt.space.Integer(low=1, high=5, name='n_layers')
nodes_space = skopt.space.Integer(low=4, high=512, name='n_nodes')
act_space = skopt.space.Categorical(categories=['relu'], name='act')

params = [learn_rate_space, layers_space, nodes_space, act_space]
params_init = [3e-5, 1, 16, 'relu']
#endregion

def log_dir_name(learn_rate, n_layers, n_nodes, act):
    log_dir = './logs/l2_{0:0e}_layers_{1}_nodes_{2}_{3}/'.format(learn_rate, n_layers, n_nodes, act)
    return log_dir

def create_model(learn_rate, n_layers, n_nodes, act):
    model = keras.Sequential()
    
    model.add(keras.layers.InputLayer(input_shape=(?)))

    for i in range(n_layers):
        model.add(keras.layers.Dense(units=n_nodes, activation=act))
    
    model.add(keras.layers.Dense(units=1, activation='linear'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

@use_named_args(dimensions=dimensions)
def fitness(learn_rate, n_layers, n_nodes, act):

    log_dir = log_dir_name(learn_rate, n_layers, n_nodes, act)

    model = create_model(learn_rate, n_layers, n_nodes, act)

    print('learning rate: {0:.1e}'.format(learn_rate))
    print('# layers:', n_layers)
    print('# nodes:', n_nodes)
    print('activation:', act)
    print()

    callback_log = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False
    )

    history = model.fit(
        x=?,
        y=?,
        batch_size=128,
        epochs=10,
        callbacks=[callback_log]
        validation_split=0.1,
        shuffle=False
    )

    acc = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print("Accuracy: {0:.2%}".format(acc))

    global best_accuracy

    if acc > acc_best:
        model.save(?)
        acc_best = acc
    
    # delete the Keras model with these hyper-parameters from memory.
    del model

    # clear the Keras session, otherwise it adds new models to preexisting graph
    K.clear_session()

    return acc

search_result = skopt.gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func='EI',
    n_calls=25,
    x0=params_init
)