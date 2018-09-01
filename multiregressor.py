import datetime as dt
import h5py
import modelib
import numpy as np
import os
import prometheus
import skopt
import tensorflow as tf
from tensorflow import keras

np.random.seed(123)
tf.set_random_seed(123)

#region Metaparams
params_search_calls = 100        # >= 11
learn_rate_space = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learn_rate')
params = [learn_rate_space]
params_init = [3e-5]
#endregion

#region Hyperparams
n_epochs = 100
batch_size = 128
split_test = 0.2
split_val = 0.25
#endregion


@skopt.utils.use_named_args(dimensions=params)
def fitness(learn_rate):
    global X_train, y_train, X_test, y_test
    try:
        X_train
    except NameError:
        X_train, y_train, X_test, y_test = prometheus.process_data()

    print('##################################')
    print('Regressor Hyperparameters:')
    print('learning rate: {}'.format(learn_rate))

    # Model instantiation
    regressor = modelib.create_model_linear(learn_rate, X_train.shape[1])

    # Run regressor
    time_start = dt.datetime.now()
    history = regressor.fit(        # train regressor
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
        validation_split=split_val,
        shuffle=False
    )
    time_end = dt.datetime.now()
    time_elapsed = time_end - time_start
    loss = history.history['val_loss'][-1]

    print('Multi-regression completed!')
    print("Validation loss: {0:.2%}".format(loss))
    print('Elapsed time: {}'.format(time_elapsed))

    update_best_regressor(regressor, loss)
    
    del regressor
    keras.backend.clear_session()

    return loss


def update_best_regressor(regressor, loss):
    global loss_best
    try:
        loss_best
    except NameError:
        loss_best = loss

    if loss <= loss_best:
        if not os.path.exists('./models/'+os.path.basename(__file__)):
            os.makedirs('./models/'+os.path.basename(__file__))

        print('New best!')
        loss_best = loss
        regressor.save_weights('./models/'+os.path.basename(__file__)+'/regressor.h5')


def read_regressor_attributes():
    '''
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze

    Author(s): Moto Mthrok, Mjshi Jewes
    '''

    weight_file_path = './models/'+os.path.basename(__file__)+'/regressor.h5'

    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print('{} contains: '.format(weight_file_path))
            print('Root attributes:')
        for key, value in f.attrs.items():
            print('  {}: {}'.format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print('  {}'.format(layer))
            print('    Attributes:')
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print('    Dataset:')
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print('      {}/{}: {}'.format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()

#endregion


if __name__ == '__main__':
    print('Commencing Multiregression optimization...')

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
    print('Multiregression optimization completed!')
    print('Results: ', params_search.space.point_to_dict(params_search.x))
    print('Fitness: ', params_search.fun)
    print('Elapsed time: {}'.format(time_elapsed))

    print('Reading regressor output...')
    read_regressor_attributes()