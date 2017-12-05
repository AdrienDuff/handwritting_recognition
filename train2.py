from modeles_training import define_and_train_model

if __name__ == "__main__":
  modele_name = "type2_CNN"
  modele_pad = "pad_white"
  modele_type = 2 # CNN type 2
  nb_exemples = None #Take all
  initial_epoch = 0
  nb_epochs = 100 # C'est trop ?
  plot_Hist = True
  display_Predict = True
  display_CER_WER = True
  define_and_train_model(modele_name, modele_pad, modele_type, nb_epochs = nb_epochs, initial_epoch = initial_epoch, \
                          optimizer = 'rmsprop', nb_exemples = nb_exemples, plotHist = plot_Hist, \
                          display_Predict = display_Predict, display_CER_WER = display_CER_WER)
