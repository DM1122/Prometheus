import distutils
import os
import shutil
import tensorflow as tf
from tensorflow import keras


def create_model_linear(rate, inputs):
    model = keras.Sequential()

    model.add(keras.layers.Dense(
        units=1,
        activation='linear',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        input_shape=(inputs,),
        name='Output'))

    opt = keras.optimizers.Adam(lr=rate)

    return model, opt


def create_model_dense(learn_rate, n_layers, n_nodes, act, n_features, dropout):        # WIP
    model = keras.Sequential()
    
    # model.add(keras.layers.InputLayer(input_shape=(n_features, ), name='Input'))

    for i in range(n_layers):
        model.add(keras.layers.Dense(
            units=n_nodes,
            activation=act,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            #kernel_regularizer=keras.regularizers.l2(l=0.00001),
            name='Dense_{}'.format(i+1)))
        
        model.add(keras.layers.Dropout(dropout))
    
    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def create_model_RNN(rate, layers, nodes, act, droprate, inputs, outputs):
    model = keras.Sequential()
    
    nodes_per_layer = nodes // layers

    model.add(keras.layers.SimpleRNN(
        units=nodes_per_layer,
        activation=act,
        dropout=droprate,
        return_sequences=True,
        input_shape=(None, inputs,),
        name='RNN_0'))

    for i in range(layers-1):
        model.add(keras.layers.SimpleRNN(
            units=nodes_per_layer,
            activation=act,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=droprate,
            recurrent_dropout=0.0,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            name='RNN_{}'.format(i+1)))
    
    model.add(keras.layers.Dense(units=outputs, activation='linear', name='Output'))

    opt = keras.optimizers.Adam(lr=rate)

    return model, opt


def create_model_LSTM(rate, layers, nodes, act, droprate, inputs, outputs):
    model = keras.Sequential()

    nodes_per_layer = nodes // layers

    model.add(keras.layers.LSTM(
        units=nodes_per_layer,
        activation=act,
        dropout=droprate,
        return_sequences=True,
        input_shape=(None, inputs,),
        name='LSTM_0'))

    for i in range(layers-1):
        model.add(keras.layers.LSTM(
            units=nodes_per_layer,
            activation=act,
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=droprate,
            recurrent_dropout=0.0,
            implementation=1,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            name='LSTM_{}'.format(i+1)))

    model.add(keras.layers.Dense(units=outputs, activation='linear', name='Output'))

    opt = keras.optimizers.Adam(lr=rate)
    
    return model, opt


def create_model_GRU(rate, layers, nodes, act, droprate, inputs, outputs):
    model = keras.Sequential()

    nodes_per_layer = nodes // layers

    model.add(keras.layers.GRU(
            units=nodes_per_layer,
            activation=act,
            dropout=droprate,
            return_sequences=True,
            input_shape=(None, inputs,),
            name='GRU_0'))

    for i in range(layers-1):        
        model.add(keras.layers.GRU(
            units=nodes_per_layer,
            activation=act,
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=droprate,
            recurrent_dropout=0.0,
            implementation=1,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            reset_after=False,
            name='GRU_{}'.format(i+1)))

    model.add(keras.layers.Dense(units=outputs, activation='linear', name='Output'))

    opt = keras.optimizers.Adam(lr=rate)
    
    return model, opt


def callbacks(log, id):
    '''
    Returns a list of configured keras callbacks.

    Args:
      log : name of individual log files
      id : script name
    '''
    
    if not os.path.exists('./logs/'+id+'/'):        # create logs dir
        os.makedirs('./logs/'+id+'/')
    
    if not os.path.exists('./models/'+id+'/'):      # create models dir
        os.makedirs('./models/'+id+'/')

    callback_NaN = keras.callbacks.TerminateOnNaN()     # NaN callback

    callback_checkpoint = keras.callbacks.ModelCheckpoint(      # checkpoint callback
        filepath='./models/'+id+'/model.keras',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)
    
    callback_early_stopping = keras.callbacks.EarlyStopping(        # early stopping callback
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto',
        baseline=None)

    callback_tensorboard = keras.callbacks.TensorBoard(     # tensorboard callback
        log_dir=log,
        histogram_freq=5,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False)

    callbacks = [callback_NaN, callback_checkpoint, callback_early_stopping, callback_tensorboard]

    return callbacks


def update_best_model(model, loss, log, id):
    '''
    Updates best model according to loss & deletes all other logs.

    Args:
      model : keras model
      loss : validation loss
      log : name of individual log file
      id : script name
    '''

    if not os.path.exists('./models/'+id+'/best/'):        # create logs dir
        os.makedirs('./models/'+id+'/best/')
    
    global loss_best
    try:
        loss_best
    except NameError:
        loss_best = loss

    if loss <= loss_best:
        print('New best!')
        loss_best = loss

        model.save('./models/'+id+'/best/model.keras')     # save best model to disk

        for l in os.listdir('./logs/'+id+'/'):
            if ('./logs/'+id+'/'+l+'/') != log:       # delete all but current prometheus_rnn log
                shutil.rmtree('./logs/'+id+'/'+l+'/')
    else:
        shutil.rmtree(log)      # delete current log