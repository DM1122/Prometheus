import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import skopt
import tensorflow as tf
from tensorflow import keras

import libs
from libs import figurelib, NNlib, NSRDBlib, processlib
import prometheus_rnn

np.random.seed(123)
tf.set_random_seed(123)


#region Metaparams
params_search_calls = 11

learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
layers_space = skopt.space.Integer(low=1, high=3, name='n_layers')
nodes_space = skopt.space.Integer(low=4, high=256, name='n_nodes')
act_space = skopt.space.Categorical(categories=['tanh'], name='act')
droprate_space = skopt.space.Real(low=0, high=0.5, name='droprate')
batch_space = skopt.space.Integer(low=1, high=256, name='batch_size')
sequence_space = skopt.space.Integer(low=1, high=336, name='sequence_length')

params = [learn_rate_space, layers_space, nodes_space, act_space, droprate_space, batch_space, sequence_space]
params_init = [1e-3, 1, 8, 'tanh', 0.0, 128, 12]
#endregion

file_name = os.path.basename(__file__)


@skopt.utils.use_named_args(dimensions=params)
def fitness(learn_rate, n_layers, n_nodes, act, droprate, batch_size, sequence_length):

    model, fitness, log = prometheus_rnn.train_model(learn_rate, n_layers, n_nodes, act, droprate, batch_size, sequence_length)

    NNlib.update_best_model(model=model, loss=fitness, log=log, logdir='./logs/prometheus_rnn.py/', id=file_name)

    del model
    keras.backend.clear_session()

    return fitness


if __name__ == '__main__':
    print('Commencing Epimetheus hyperparameter optimization...')

    #region Hyperparameter optimization
    time_start = dt.datetime.now()

    params_search = skopt.gp_minimize(      # bayesian optimization
        func=fitness,
        dimensions=params,
        acq_func='EI',
        n_calls=params_search_calls,
        x0=params_init
    )

    time_end = dt.datetime.now()
    time_elapsed = time_end - time_start

    print('#################################################')
    print('')
    print('Hyperparameter optimization completed!')
    print('Results: ', params_search.space.point_to_dict(params_search.x))
    print('Fitness: ', params_search.fun)
    print('Elapsed time: ', time_elapsed)
    NNlib.log_search(
        search=params_search.space.point_to_dict(params_search.x),
        score=params_search.fun,
        time=time_elapsed,
        date=dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
        id=file_name)
    #endregion

    #region Plots
    fig1, fig2, fig3 = figurelib.plot_opt(search=params_search, dims=['learn_rate', 'n_layers', 'n_nodes', 'droprate', 'batch_size', 'sequence_length'])

    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    figurelib.save_fig(fig=fig1, name='convergence_plot.png', date=plot_date, id=file_name)
    figurelib.save_fig(fig=fig2, name='objective_plot.png', date=plot_date, id=file_name)
    figurelib.save_fig(fig=fig3, name='evaluations_plot.png', date=plot_date, id=file_name)

    plt.show()
    #endregion

