import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow import keras


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


def update_best_model(model, loss, log, logdir, id):
    '''
    Updates best model according to loss & deletes all other logs.

    Args:
      model : keras model
      loss : validation loss
      log : name of individual log file
      logdir : directory of stored logs
      id : script name
    '''

    if not os.path.exists('./models/'+id+'/best/'):
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

        for l in os.listdir(logdir):
            if (logdir+l+'/') != log:       # delete all but current log from logdir
                shutil.rmtree(logdir+l+'/')
    else:
        shutil.rmtree(log)      # delete current log


def batch_generator(x, y, batch_size):
    '''
    Generator function for creating random batches of data.

    Author: Hvass-Labs
    Modifier: DM1122

    Args:
      x : feature dataset
      y : label dataset
      batch_size : number of samples
    '''

    # Infinite loop
    while True:
        # Allocate new empty array for batch of input signals
        x_batch = np.zeros(shape=(batch_size, x.shape[1]))

        # Allocate new empty array for batch of output signals
        y_batch = np.zeros(shape=(batch_size, y.shape[1]))

        # Fill batch arrays with data
        for i in range(batch_size):
            # Get random start index
            idx = np.random.randint(x.shape[0] - batch_size)
            
            # Copy the sequences of data starting at idx
            x_batch = x[idx:idx+batch_size]
            y_batch = y[idx:idx+batch_size]
        
        yield (x_batch, y_batch)


def batch_generator_seq(x, y, batch_size, timesteps):
    '''
    Generator function for creating random batches of sequential data.

    Author: Hvass-Labs
    Modifier: DM1122

    Args:
      x : feature dataset
      y : label dataset
      batch_size : number of samples
      timesteps : sequence length
    '''

    # Infinite loop
    while True:
        # Allocate new empty array for batch of input signals
        x_batch = np.zeros(shape=(batch_size, timesteps, x.shape[1]))

        # Allocate new empty array for batch of output signals
        y_batch = np.zeros(shape=(batch_size, timesteps, y.shape[1]))

        # Fill batch arrays with random sequences of data
        for i in range(batch_size):
            # Get random start index
            idx = np.random.randint(x.shape[0] - timesteps)
            
            # Copy the sequences of data starting at idx
            x_batch[i] = x[idx:idx+timesteps]
            y_batch[i] = y[idx:idx+timesteps]
        
        yield (x_batch, y_batch)


def calculate_loss_warmup(y_true, y_pred):     # WIP
    '''
    Calculate the MSE between y_true and y_pred,
    ignoring the beginning "warmup" part of the sequences.

    Author: Hvass-Labs
    Modifier: DM1122

    Args:
      y_true : desired output
      y_pred : model output
      warmup (internal) : length of sequence to ignore
    '''


    warmup_length = 84

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    # y_true_slice = y_true[:, warmup:, :]
    # y_pred_slice = y_pred[:, warmup:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    # print(y_true.shape)
    # print(y_pred.shape)

    # if (features_label_scale):      # unscale datasets (does not work)

    #     y_true = y_scl.inverse_transform(y_test)
    #     y_pred = y_scl.inverse_transform(y_pred)

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_warmup = tf.reduce_mean(loss)

    return loss_warmup


def log_search(search, score, time, date, id):
    '''
    Logs hyperparameter search results to .txt file.

    Args:
      search : parameter search dict
      score : search fitness
      time : elapsed search time
      date : formatted timestamp
      id : script name
    '''

    if not os.path.exists('./logs/'+id+'/'):
        os.makedirs('./logs/'+id+'/')

    with open(file='./logs/'+id+'/'+date+'.txt', mode='w') as txt:
        txt.write('Hyperparameter optimization log:\n')
        txt.write('--------------------------------\n')
        txt.write('Results: {}\n'.format(search))
        txt.write('Fitness: {}\n'.format(score))
        txt.write('Elapsed time: {}\n'.format(time))