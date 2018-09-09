import tensorflow as tf
from tensorflow import keras


#region Functions
def create_model_linear(learn_rate, n_features):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(n_features, ), name='Input'))
    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def create_model_dense(learn_rate, n_layers, n_nodes, act, n_features):
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
    
    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def create_model_RNN(learn_rate, n_layers, n_nodes, act, n_features, sequence_length):
    model = keras.Sequential()
    
    # model.add(keras.layers.InputLayer(input_shape=(sequence_length, n_features, )))

    for i in range(n_layers):
        model.add(keras.layers.SimpleRNN(units=n_nodes, activation=act, return_sequences=True, name='RNN_{}'.format(i+1)))
    
    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def create_model_LSTM(learn_rate, n_layers, n_nodes, act):
    model = keras.Sequential()

    for i in range(n_layers):
        model.add(keras.layers.LSTM(units=n_nodes, activation=act, recurrent_activation='hard_sigmoid', return_sequences=True, name='LSTM_{}'.format(i+1)))

    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def create_model_GRU(learn_rate, n_layers, n_nodes, act):
    model = keras.Sequential()

    for i in range(n_layers):
        model.add(keras.layers.GRU(units=n_nodes, activation=act, recurrent_activation='hard_sigmoid', return_sequences=True, name='GRU_{}'.format(i+1)))

    model.add(keras.layers.Dense(units=1, activation='linear', name='Output'))

    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
#endregion