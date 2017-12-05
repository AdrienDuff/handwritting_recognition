import numpy as np
from utils import load_data_white_pad, load_data_null_pad, eval_CER_and_WER
from modeles_definition import run_CTCModel, plot_history

def define_and_train_model(modele_name, modele_pad, modele_type, nb_epochs = 10, initial_epoch = 0, optimizer = 'rmsprop', \
                            nb_exemples = None, plotHist = True, display_Predict = True, display_CER_WER = True):
  if modele_pad == "pad_null":
    (x_train_pad, y_train_pad), (x_valid_pad, y_valid_pad), all_lab_train = load_data_null_pad()
  elif modele_pad == "pad_white":
    (x_train_pad, y_train_pad), (x_valid_pad, y_valid_pad), all_lab_train = load_data_white_pad()

  if nb_exemples != None:
    x_train_pad = x_train_pad[:nb_exemples]
    y_train_pad = y_train_pad[:nb_exemples]

  nb_classes = len(all_lab_train)
  nb_train = len(x_train_pad)
  nb_valid = len(x_valid_pad)

  print(x_train_pad[0].shape)
  print(x_valid_pad[0].shape)
  print(y_train_pad[0].shape)
  print(y_valid_pad[0].shape)

  x_train_len = np.asarray([len(x_train_pad[i]) for i in range(nb_train)])
  x_valid_len = np.asarray([len(x_valid_pad[i]) for i in range(nb_valid)])

  y_train_len = np.asarray([len(y_train_pad[i]) for i in range(nb_train)])
  y_valid_len = np.asarray([len(y_valid_pad[i]) for i in range(nb_valid)])


  network, history = run_CTCModel(modele_type, x_train_pad, y_train_pad, x_train_len, y_train_len, nb_epochs, initial_epoch, \
    nb_classes, modele_name, validation_data = ([x_valid_pad, y_valid_pad, x_valid_len, y_valid_len],np.zeros(nb_valid)))



  if display_Predict:
    pred_valid = network.predict2([x_valid_pad, x_valid_len])
    # Print prediction and true labeling
    for i in range(len(pred_valid)):
      print("Prediction :", pred_valid[i], " -- Label : ", y_valid_pad[i])

  if display_CER_WER:
    [cer, wer] = eval_CER_and_WER(pred_valid, y_valid_pad)

  if plotHist:
    plot_history(history, modele_name, initial_epoch, nb_epochs)


if __name__ == "__main__":
  """
  modele_name = "essaie4_test_BLSTM"
  modele_pad = "pad_null"
  modele_type = 1 # BLTSM
  nb_exemples = 128
  initial_epoch = 0
  nb_epochs = 2
  plot_Hist = True
  display_Predict = True
  display_CER_WER = True
  define_and_train_model(modele_name, modele_pad, modele_type, nb_epochs = nb_epochs, initial_epoch = initial_epoch, \
                          optimizer = 'rmsprop', nb_exemples = nb_exemples, plotHist = True, display_Predict = True, \
                          display_CER_WER = True)

  """
  modele_name = "essaie2_test_CNN"
  modele_pad = "pad_white"
  modele_type = 2 # CNN
  nb_exemples = None
  initial_epoch = 0
  nb_epochs = 3
  plot_Hist = True
  display_Predict = True
  display_CER_WER = True
  define_and_train_model(modele_name, modele_pad, modele_type, nb_epochs = nb_epochs, initial_epoch = initial_epoch, \
                          optimizer = 'rmsprop', nb_exemples = nb_exemples, plotHist = plot_Hist, \
                          display_Predict = display_Predict, display_CER_WER = display_CER_WER)

