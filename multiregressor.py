import datetime as dt
import libs
from libs import figurelib, modelib, NSRDBlib, processlib
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import skopt
import tensorflow as tf
from tensorflow import keras

np.random.seed(123)
tf.set_random_seed(123)


#region Metaparams
params_search_calls = 100

learn_rate_space = skopt.space.Real(low=1e-10, high=1e-0, prior='log-uniform', name='learn_rate')

params = [learn_rate_space]
params_init = [1e-3]
#endregion

#region Hyperparams
n_epochs = 100
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
    'pressure',
    'humidity_rel',
    'albedo_sur',
    'wind_speed',
    'wind_dir',
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
features_label_shift = 12
features_label_scale = False
features_dropzeros = True

valid_split = 0.2
test_split = 0.2
#endregion

file_name = os.path.basename(__file__)


@skopt.utils.use_named_args(dimensions=params)
def fitness(learn_rate):
    global call_count       # keep track of hyperparam search calls
    try:
        call_count
    except NameError:
        call_count = 0

    #region Model instantiation
    regressor, opt = modelib.create_model_linear(rate=learn_rate, inputs=X_train.shape[1])
    regressor.compile(optimizer=opt, loss='mse', metrics=['mae'])

    log_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = './logs/'+file_name+'/{0}_rate({1})/'.format(log_date, learn_rate)
    callbacks = modelib.callbacks(log=log_dir, id=file_name)
    #endregion

    #region Training
    print('=================================================')
    print('')
    print('Regressor Hyperparameters:')
    print('learning rate: ', learn_rate)

    time_start = dt.datetime.now()

    regressor.fit(
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


    time_end = dt.datetime.now()
    time_elapsed = time_end - time_start
    call_count += 1

    print('Multiregressor training completed!')
    #endregion

    #region Validation
    regressor.load_weights('./models/'+file_name+'/model.keras')        # restore best weights
    
    result = regressor.evaluate(x=X_valid, y=y_valid, batch_size=None, verbose=0, sample_weight=None, steps=None)

    for metric, val in zip(regressor.metrics_names, result):
        if metric == 'loss':
            loss = val

    print('Validation loss: ', loss)
    print('Elapsed time: ', time_elapsed)
    print('Search {0}/{1}'.format(call_count, params_search_calls))

    modelib.update_best_model(model=regressor, loss=loss, log=log_dir, id=file_name)
    #endregion

    del regressor
    keras.backend.clear_session()

    return loss


if __name__ == '__main__':
    print('Commencing Multiregressor optimization...')

    #region Data handling
    data = NSRDBlib.get_data(features)
    X_train, y_train, X_valid, y_valid, _, _, _ = processlib.process(
        data=data,
        label=features_label,
        shift=features_label_shift,
        dropzeros=features_dropzeros,
        labelscl=features_label_scale,
        split_valid=valid_split,
        split_test=test_split)
    #endregion

    #region Hyperparameter optimization
    time_opt_start = dt.datetime.now()

    params_search = skopt.gp_minimize(
        func=fitness,
        dimensions=params,
        acq_func='EI',
        n_calls=params_search_calls,
        x0=params_init
    )

    time_opt_end = dt.datetime.now()
    time_opt_elapsed = time_opt_end - time_opt_start

    print('#################################################')
    print('')
    print('Multiregressor optimization completed!')
    print('Results: ', params_search.space.point_to_dict(params_search.x))
    print('Fitness: ', params_search.fun)
    print('Elapsed time: ', time_opt_elapsed)
    #endregion

    #region Figures
    fig1, fig2, fig3 = figurelib.plot_opt_single(search=params_search, dim='learn_rate')

    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    figurelib.save_fig(fig=fig1, name='convergence_plot.png', date=plot_date, id=file_name)
    figurelib.save_fig(fig=fig2, name='objective_plot.png', date=plot_date, id=file_name)
    figurelib.save_fig(fig=fig3, name='evaluations_plot.png', date=plot_date, id=file_name)

    plt.show()
    #endregion

    #region Weights
    print('Reading model weights...')

    model = keras.models.load_model('./models/'+file_name+'/best/model.keras')

    for layer in model.layers:
        print(layer.get_config())
        print(layer.get_weights())
    #endregion
