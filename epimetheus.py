import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import prometheus
import shutil
import skopt
import tensorflow as tf
from tensorflow import keras

np.random.seed(123)
tf.set_random_seed(123)


#region Metaparams
params_search_calls = 100       # must be >=11 (risk 'The objective has been evaluated at this point before' w/ values >100)

learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
layers_space = skopt.space.Integer(low=1, high=3, name='n_layers')
nodes_space = skopt.space.Integer(low=2, high=1024, name='n_nodes')
act_space = skopt.space.Categorical(categories=['relu'], name='act')
dropout_space = skopt.space.Real(low=0, high=0.5, name='dropout')

params = [learn_rate_space, layers_space, nodes_space, act_space, dropout_space]
params_init = [3e-5, 1, 16, 'relu', 0.2]
#endregion


#region Functions
@skopt.utils.use_named_args(dimensions=params)      # allows params to be passed as list
def fitness(learn_rate, n_layers, n_nodes, act, dropout):

    model, fitness, log_dir = prometheus.train_model(learn_rate, n_layers, n_nodes, act, dropout)

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

    # determine best model
    if loss <= loss_best:
        if not os.path.exists('./models/'+os.path.basename(__file__)):      # model.save() does not explicitly create dir
            os.makedirs('./models/'+os.path.basename(__file__))

        print('New best!')
        loss_best = loss      # update best accuracy
        model.save('./models/'+os.path.basename(__file__)+'/model.keras')     # save best model to disk

        # delete all other model logs
        for log in os.listdir('./logs/'):
            if ('./logs/'+log+'/') != log_dir:
                shutil.rmtree('./logs/'+log+'/')
    else:
        shutil.rmtree(log_dir)      # delete current log


def plot_results(search):
    matplotlib.style.use('classic')
    fig1 = skopt.plots.plot_convergence(search)
    fig2, ax = skopt.plots.plot_histogram(result=search, dimension_name='n_layers')
    fig3, ax = skopt.plots.plot_objective(result=search, dimension_names=['learn_rate','n_layers','n_nodes'])
    fig4, ax = skopt.plots.plot_evaluations(result=search, dimension_names=['learn_rate','n_layers','n_nodes'])
    plt.show()
#endregion


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

    print('#########################################')
    print('Hyperparameter optimization completed!')
    print('Results: ', params_search.space.point_to_dict(params_search.x))
    print('Fitness: ', params_search.fun)
    print('Elapsed time: {}'.format(time_elapsed))
    plot_results(params_search)

    print('Debug:\n$tensorboard --logdir=logs')


