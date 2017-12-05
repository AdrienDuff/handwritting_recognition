import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, GRU, Activation, TimeDistributed, Lambda, Masking, Bidirectional, GaussianNoise
from keras.optimizers import RMSprop, Adam
from keras.engine import Model

import numpy as np
from random import randint
import pickle
from CTCModel import CTCModel


from keras.preprocessing import sequence




def rnn_seqlab_network(dim1, dim2, nb_labels, prob_dropout=0.1):
    """ Recurrent Neural Network for sequence analysis """

    inputs = Input(name='input', shape=[dim1, dim2])
    layer = Masking(mask_value=255, input_shape=(None, None))(inputs)
    layer = GaussianNoise(0.01)(layer)
    layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
    layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
    #layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
    layer = TimeDistributed(Dense(nb_labels+1, name="dense"))(layer)
    predictions = Activation('softmax', name='softmax')(layer)

    network = CTCModel([inputs], [predictions])
    network.compile('rmsprop')

    return network


def run_CTCModel(x_train_pad, y_train_pad, x_train_len, y_train_len, nb_epochs, initial_epoch, nb_classes):

    nb_train = len(x_train_len)
    d2 = x_train_pad.shape[-1]

    # Save the weigths of the model
    callbacks = [ModelCheckpoint("./model/" + "weights.{epoch:02d}.hdf5", period=nb_epochs)]

    # create the structure of the model or load it + the weights at epoch 'initial_epoch'
    if initial_epoch == 0:
        network = rnn_seqlab_network(None, d2, nb_classes, prob_dropout=0.1)
        network.save_model('./model/')
    else:
        network = CTCModel(None, None)
        network.load_model('./model/', 'rmsprop', initial_epoch)

    network.fit(x=[x_train_pad, y_train_pad, x_train_len, y_train_len], y=np.zeros(nb_train), \
                batch_size=32, callbacks=callbacks, epochs=nb_epochs, initial_epoch=initial_epoch)
    pred = network.predict2([x_test_pad, x_test_len])

    # Print prediction and true labeling
    for i in range(10):  # len(pred)):
        print("Prediction :", pred[i], " -- Label : ", y_test[i])



if __name__ == '__main__':


    # load generated data
    (x_train, y_train), (x_test, y_test) = pickle.load(open('./seqDigitsVar.pkl', 'rb'))
    nb_train = len(x_train)
    nb_test = len(x_test)
    #d2 = len(x_train[0][0])
    nb_classes= 10




    x_train_pad = sequence.pad_sequences(x_train, value=float(255), dtype='float32',
                                  padding="post", truncating='post')
    x_test_pad = sequence.pad_sequences(x_test, value=float(255), dtype='float32',
                                     padding="post", truncating='post')

    y_train_pad = sequence.pad_sequences(y_train, value=float(10), maxlen=10,
                                    dtype='float32', padding="post")
    y_test_pad = sequence.pad_sequences(y_test, value=float(10), maxlen=10,
                                     dtype='float32', padding="post")



    x_train_len = np.asarray([len(x_train[i]) for i in range(nb_train)])
    x_test_len = np.asarray([len(x_test[i]) for i in range(nb_test)])
    y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])


    # FIRST TRAINING
    nb_epochs = 2
    initial_epoch = 0

    run_CTCModel(x_train_pad, y_train_pad, x_train_len, y_train_len, nb_epochs, initial_epoch, nb_classes)

    # SECOND TRAINING USING THE WEIGHTS THAT HAVE BEEN SAVED
    nb_epochs = 4
    initial_epoch = 2

    run_CTCModel(x_train_pad, y_train_pad, x_train_len, y_train_len, nb_epochs, initial_epoch, nb_classes)