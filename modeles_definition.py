import numpy as np
from utils import load_data_white_pad, load_data_null_pad, eval_CER_and_WER

import keras
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
from CTCModel import CTCModel

from keras.layers import Dense, Input, LSTM, Activation, TimeDistributed, Masking, \
             Bidirectional, GaussianNoise, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Dropout, Permute, Reshape, Flatten
from keras.optimizers import RMSprop, Adam
from keras.engine import Model
import os
import shutil

import matplotlib.pyplot as plt

def rnn_seqlab_networkBLSTM1(dim1, dim2, nb_labels, prob_dropout=0.1):
  """ Recurrent Neural Network for sequence analysis """

  inputs = Input(name='input', shape=[dim1, dim2])
  layer = Masking(mask_value=-1, input_shape=(None, None))(inputs)
  layer = GaussianNoise(0.01)(layer)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  #layer = LSTM(1, return_sequences=True, dropout=prob_dropout)(layer)
  #layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  layer = TimeDistributed(Dense(nb_labels+1, name="dense"))(layer)
  predictions = Activation('softmax', name='softmax')(layer)

  network = CTCModel([inputs], [predictions])
  network.compile('rmsprop')
  return network

def rnn_seqlab_networkCNN0(dim1, dim2, nb_labels, prob_dropout=0.1):
  """ Recurrent Neural Network for sequence analysis """

  inputs = Input(name='input', shape=[dim1, dim2])
  #layer = Masking(mask_value=255, input_shape=(None, None,1))(inputs)
  #layer = Masking(mask_value=255)(inputs)
  layer = GaussianNoise(0.01)(inputs)
  print(layer._keras_shape)
  layer = Reshape((dim1,dim2,1))(layer)
  print(layer._keras_shape)
  layer = TimeDistributed(Conv1D(filters = 8, kernel_size = 3 , activation ='relu'))(layer)
  print(layer._keras_shape)
  layer = TimeDistributed(Dropout(prob_dropout))(layer)
  print(layer._keras_shape)
  layer = TimeDistributed(MaxPooling1D(pool_size = 2))(layer)
  print(layer._keras_shape)
  layer = TimeDistributed(Flatten())(layer)
  print(layer._keras_shape)
  #layer = Permute((2, 1))(layer)
  layer = Bidirectional(LSTM(25, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  #layer = LSTM(1, return_sequences=True, dropout=prob_dropout)(layer)
  #layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  layer = TimeDistributed(Dense(nb_labels+1, name="dense"))(layer)
  print(layer._keras_shape)
  predictions = Activation('softmax', name='softmax')(layer)

  network = CTCModel([inputs], [predictions])
  network.compile('rmsprop')
  return network


def rnn_seqlab_networkCNN2(dim1, dim2, nb_labels, prob_dropout=0.1):
  """ Recurrent Neural Network for sequence analysis """

  inputs = Input(name='input', shape=[dim1, dim2])
  #layer = Masking(mask_value=255, input_shape=(None, None,1))(inputs)
  #layer = Masking(mask_value=255)(inputs)
  layer = GaussianNoise(0.01)(inputs)

  layer = Reshape((dim1,dim2,1))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 8, kernel_size = 4 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 4, kernel_size = 4 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Reshape((334,8*4))(layer)
  print(layer._keras_shape)
  #layer = Permute((2, 1))(layer)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  #layer = LSTM(1, return_sequences=True, dropout=prob_dropout)(layer)
  #layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  layer = TimeDistributed(Dense(nb_labels+1, name="dense"))(layer)
  print(layer._keras_shape)
  predictions = Activation('softmax', name='softmax')(layer)

  network = CTCModel([inputs], [predictions])
  network.compile('rmsprop')
  return network

def rnn_seqlab_networkCNN3(dim1, dim2, nb_labels, prob_dropout=0.1):
  """ Recurrent Neural Network for sequence analysis """

  inputs = Input(name='input', shape=[dim1, dim2])
  #layer = Masking(mask_value=255, input_shape=(None, None,1))(inputs)
  #layer = Masking(mask_value=255)(inputs)
  layer = GaussianNoise(0.01)(inputs)

  layer = Reshape((dim1,dim2,1))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 16, kernel_size = 3 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 32, kernel_size = 3 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Reshape((334,8*32))(layer)
  print(layer._keras_shape)
  #layer = Permute((2, 1))(layer)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  #layer = LSTM(1, return_sequences=True, dropout=prob_dropout)(layer)
  #layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  layer = TimeDistributed(Dense(nb_labels+1, name="dense"))(layer)
  print(layer._keras_shape)
  predictions = Activation('softmax', name='softmax')(layer)

  network = CTCModel([inputs], [predictions])
  network.compile('rmsprop')
  return network


def rnn_seqlab_networkCNN4(dim1, dim2, nb_labels, prob_dropout=0.1):
  """ Recurrent Neural Network for sequence analysis """

  inputs = Input(name='input', shape=[dim1, dim2])
  #layer = Masking(mask_value=255, input_shape=(None, None,1))(inputs)
  #layer = Masking(mask_value=255)(inputs)
  layer = GaussianNoise(0.01)(inputs)

  layer = Reshape((dim1,dim2,1))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 16, kernel_size = 3 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 32, kernel_size = 3 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Conv2D(filters = 64, kernel_size = 3 , padding = "same", activation ='relu')(layer)
  print(layer._keras_shape)
  layer = Dropout(prob_dropout)(layer)
  print(layer._keras_shape)
  layer = MaxPooling2D(pool_size = (1,2))(layer)
  print(layer._keras_shape)
  layer = Reshape((334,4*64))(layer)
  print(layer._keras_shape)
  #layer = Permute((2, 1))(layer)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  print(layer._keras_shape)
  #layer = LSTM(1, return_sequences=True, dropout=prob_dropout)(layer)
  #layer = Bidirectional(LSTM(50, return_sequences=True, dropout=prob_dropout))(layer)
  layer = TimeDistributed(Dense(nb_labels+1, name="dense"))(layer)
  print(layer._keras_shape)
  predictions = Activation('softmax', name='softmax')(layer)

  network = CTCModel([inputs], [predictions])
  network.compile('rmsprop')
  return network


def run_CTCModel(model_type, x_train_pad, y_train_pad, x_train_len, y_train_len, nb_epochs, initial_epoch, nb_classes, modele_name, validation_data = None):
  nb_train = len(x_train_len)
  d1, d2 = x_train_pad[0].shape

  # Save the weigths of the model
  #callbacks = [ModelCheckpoint("./Modeles/" + modele_name + "/weights.{epoch:02d}.hdf5", period=nb_epochs)]
  callbacks = [ModelCheckpoint("./Modeles/" + modele_name + "/weights.{epoch:02d}.hdf5", period=1,save_best_only=True), \
              EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')]

  # create the structure of the model or load it + the weights at epoch 'initial_epoch'. Manage a file system.
  if initial_epoch == 0:
    if model_type == 1:
      network = rnn_seqlab_networkBLSTM1(None, d2, nb_classes, prob_dropout=0.1)
    if model_type == 2:
      network = rnn_seqlab_networkCNN2(d1, d2, nb_classes, prob_dropout=0.1)
    if model_type == 3:
      network = rnn_seqlab_networkCNN3(d1, d2, nb_classes, prob_dropout=0.1)
    if model_type == 4:
      network = rnn_seqlab_networkCNN4(d1, d2, nb_classes, prob_dropout=0.1)
    if os.path.isdir('Modeles/' + modele_name):
      #On vide le dossier
      shutil.rmtree('Modeles/' + modele_name)
      os.mkdir('Modeles/' + modele_name)
      print('\n----- Ecrasement + Création d\'un nouveau modèle -----\n')
    else:
      try:
        os.mkdir('Modeles/' + modele_name)
        print('\n----- Création d\'un nouveau modèle -----\n')
      except OSError:
        print('\n----- Erreur dans la création du dossier -----\n')
    network.save_model('./Modeles/' + modele_name)
  else:
    network = CTCModel(None, None)
    network.load_model('Modeles/' + modele_name + '/', optimizer, initial_epoch)
    print('\n----- Chargement du modèle -----\n')




  history = network.fit(x=[x_train_pad, y_train_pad, x_train_len, y_train_len], y=np.zeros(nb_train), \
              batch_size=32, callbacks=callbacks, epochs=nb_epochs, initial_epoch=initial_epoch, validation_data = validation_data)
  return network, history


def plot_history(history, modele_name, initial_epoch, nb_epochs):
  plt.plot(np.log(history.history['loss']))
  plt.plot(np.log(history.history['val_loss']))
  plt.title('model loss')
  plt.ylabel('log loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig('./Modeles/' + modele_name + '/loss_' + str(initial_epoch) + '-' + str(nb_epochs) + '.png')


if __name__ == '__main__':
  optimizer = 'rmsprop'
  nb_exemples = 128
  #(x_train_pad, y_train_pad), (x_valid_pad, y_valid_pad), all_lab_train = load_data_white_pad()

  (x_train_pad, y_train_pad), (x_valid_pad, y_valid_pad), all_lab_train = load_data_null_pad()


  x_train_pad = x_train_pad[:nb_exemples]
  y_train_pad = y_train_pad[:nb_exemples]
  nb_classes = len(all_lab_train)
  nb_train = len(x_train_pad)
  nb_valid = len(x_valid_pad)

  _ , D = x_train_pad[0].shape # D = 35 pixel de large, la bande est variable en longeur


  x_train_len = np.asarray([len(x_train_pad[i]) for i in range(nb_train)])
  x_valid_len = np.asarray([len(x_valid_pad[i]) for i in range(nb_valid)])

  y_train_len = np.asarray([len(y_train_pad[i]) for i in range(nb_train)])
  y_valid_len = np.asarray([len(y_valid_pad[i]) for i in range(nb_valid)])


  modele_name = "essaie2_test_BLSTM"
  initial_epoch = 0
  nb_epochs = 6
  network, history = run_CTCModel(1,x_train_pad, y_train_pad, x_train_len, y_train_len, nb_epochs, initial_epoch, \
    nb_classes, modele_name, validation_data = ([x_valid_pad, y_valid_pad, x_valid_len, y_valid_len],np.zeros(nb_valid)))


  pred_valid = network.predict2([x_valid_pad, x_valid_len])

  # Print prediction and true labeling
  for i in range(len(pred_valid)):
    print("Prediction :", pred_valid[i], " -- Label : ", y_valid_pad[i])

  [cer, wer] = eval_CER_and_WER(pred_valid, y_valid_pad)
  plot_history(history, modele_name, initial_epoch, nb_epochs)
