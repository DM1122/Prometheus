import datetime as dt
import distutils
from distutils.dir_util import copy_tree
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import prometheus_rnn
import shutil
import skopt
import tensorflow as tf
from tensorflow import keras

np.random.seed(123)
tf.set_random_seed(123)


#region Metaparams
params_search_calls = 11

learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
layers_space = skopt.space.Integer(low=1, high=3, name='n_layers')
nodes_space = skopt.space.Integer(low=2, high=256, name='n_nodes')
act_space = skopt.space.Categorical(categories=['tanh','sigmoid'], name='act')
droprate_space = skopt.space.Real(low=0, high=0.5, name='droprate')
batch_space = skopt.space.Integer(low=1, high=256, name='batch_size')
sequence_space = skopt.space.Integer(low=1, high=336, name='sequence_length')

params = [learn_rate_space, layers_space, nodes_space, act_space, droprate_space, batch_space, sequence_space]
params_init = [3e-5, 1, 8, 'relu', 0.2, 256, 334]
#endregion


@skopt.utils.use_named_args(dimensions=params)
def fitness(learn_rate, n_layers, n_nodes, act, droprate, batch_size, sequence_length):

    model, fitness, log_dir = prometheus_rnn.train_model(learn_rate, n_layers, n_nodes, act, droprate, batch_size, sequence_length)

    update_best_model(model, fitness, log_dir)

    del model
    keras.backend.clear_session()

    return fitness


def update_best_model(model, loss, log_dir):
    global loss_best
    try:        # will set best loss to current loss on first run
        loss_best
    except NameError:
        loss_best = loss

    if not os.path.exists('./models/epimetheus_rnn/'):
        os.makedirs('./models/epimetheus_rnn/')
    if not os.path.exists('./logs/epimetheus_rnn/'):
        os.makedirs('./logs/epimetheus_rnn/')


    if loss <= loss_best:
        print('New best!')
        loss_best = loss      # update best accuracy

        model.save('./models/epimetheus_rnn/model.keras')     # save best model to disk

        for log in os.listdir('./logs/prometheus_rnn/'):        
            if ('./logs/prometheus_rnn/'+log+'/') == log_dir:
                copy_tree('./logs/prometheus_rnn/'+log+'/', './logs/epimetheus_rnn/'+log+'/')       # copy log to epimetheus logbase

            if ('./logs/prometheus_rnn/'+log+'/') != log_dir:       # delete all but current prometheus_rnn log
                shutil.rmtree('./logs/prometheus_rnn/'+log+'/')
    else:
        shutil.rmtree(log_dir)      # delete current log


def plot_results(search):
    matplotlib.style.use('classic')

    fig1 = skopt.plots.plot_convergence(search)
    fig2, ax = skopt.plots.plot_objective(result=search, dimension_names=['learn_rate','n_layers','n_nodes','droprate','batch_size','sequence_length'])
    fig3, ax = skopt.plots.plot_evaluations(result=search, dimension_names=['learn_rate','n_layers','n_nodes','droprate','batch_size','sequence_length'])

    # Save plots to disk
    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    plot_dir = './plots/epimetheus_rnn/'+plot_date+'/'
    os.makedirs(plot_dir)
    # fig1.savefig(plot_dir+'convergence_plot.png')      # does not work, must save manually!
    fig2.savefig(plot_dir+'objective_plot.png')
    fig3.savefig(plot_dir+'evaluations_plot.png')

    plt.show()

if __name__ == '__main__':
    print('Commencing Epimetheus hyperparameter optimization...')

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
    print('Elapsed time: {}'.format(time_elapsed))
    plot_results(params_search)