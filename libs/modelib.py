import tensorflow as tf
from tensorflow import keras


def create_model_linear(learn_rate, n_features):        # WIP
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(n_features, ), name='Input'))
    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


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