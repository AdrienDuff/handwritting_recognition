from PIL import Image
import numpy as np
import operator
import pickle

import keras
import keras.backend as K
import tensorflow as tf
from keras.preprocessing import sequence

def read_data(my_directory, set_to_read='train'):
    """
    :param my_directory:
    :param set_to_read: dataset to load
    :return:
    list_im = list of images that have been resized
    list_labels = list of string
    all_labels = dictionary with the labels and the refering index

    """

    list_im = []
    list_labels = []
    with open(my_directory + set_to_read + '_labels.txt', 'r', encoding="ISO-8859-1") as f:

        for line in f.readlines():
            sline = line.split(' ')
            name_file = sline[0]
            label = ' '.join(sline[1:]).replace('\n', '')

            image = Image.open(my_directory + name_file)
            image = image.convert('L')
            width, height = image.size
            ratio = 35. / height
            image = image.resize((int(width * ratio), 35), Image.ANTIALIAS)
            im = np.transpose(np.array(image))
            list_im.append(im)
            list_labels.append(label)


    if set_to_read == 'train':
        k=0
        all_lab = {}
        for elmt in list_labels:

            for ch in elmt:
                if ch not in all_lab:
                    all_lab[ch] = k
                    k += 1
    else:
        all_lab = {}

    return list_im, list_labels, all_lab



def convert_ch_to_int(y, all_lab):
    """ convert a character to a string from the dictionary all_lab"""

    out = []
    for elmt in y:
        ch_int = []
        for ch in elmt:
            ch_int.append(all_lab[ch])

        out.append(ch_int)

    return out

def load_data_raw(set_to_read = 'train'):
  try:
    with open('data/' + set_to_read + '_pickle', 'rb') as my_pickle:
      list_im, list_labels, all_lab = pickle.load(my_pickle)
  except IOError:
    list_im, list_labels, all_lab = read_data('data/', set_to_read=set_to_read)
    pickle.dump([list_im, list_labels, all_lab], open( 'data/' + set_to_read + '_pickle', 'wb' ))
  return list_im, list_labels, all_lab


def load_data_white_pad():
    list_im_train, list_labels_train, all_lab_train = load_data_raw(set_to_read = 'train')
    list_im_valid, list_labels_valid, _ = load_data_raw(set_to_read = 'valid')

    try:
        with open('data/all_data_white_pad_pickle', 'rb') as my_pickle:
            (x_train_pad, y_train_conv_pad), (x_valid_pad, y_valid_conv_pad) = pickle.load(my_pickle)
    except IOError:
        nb_classes = len(all_lab_train)
        x_train_pad = sequence.pad_sequences(list_im_train, value=float(255), dtype='float32',
                                    padding="post", truncating='post')
        pad_value, _ = x_train_pad[0].shape
        print("pad value = " + str(pad_value))
        x_valid_pad = sequence.pad_sequences(list_im_valid, value=float(255), dtype='float32',
                                     padding="post", truncating='post', maxlen = pad_value)

        list_labels_train_conv = convert_ch_to_int(list_labels_train, all_lab_train)
        list_labels_valid_conv = convert_ch_to_int(list_labels_valid, all_lab_train)

        y_train_conv_pad = sequence.pad_sequences(list_labels_train_conv, value=float(nb_classes + 1),
                                    dtype='float32', padding="post")
        y_valid_conv_pad = sequence.pad_sequences(list_labels_valid_conv, value=float(nb_classes + 1),
                                     dtype='float32', padding="post")
        pickle.dump([(x_train_pad, y_train_conv_pad), (x_valid_pad, y_valid_conv_pad)],
                    open( 'data/all_data_white_pad_pickle', 'wb' ))
    return (x_train_pad, y_train_conv_pad), (x_valid_pad, y_valid_conv_pad), all_lab_train

def load_data_null_pad():
    list_im_train, list_labels_train, all_lab_train = load_data_raw(set_to_read = 'train')
    list_im_valid, list_labels_valid, _ = load_data_raw(set_to_read = 'valid')

    try:
        with open('data/all_data_null_pad_pickle', 'rb') as my_pickle:
            (x_train_pad, y_train_conv_pad), (x_valid_pad, y_valid_conv_pad) = pickle.load(my_pickle)
    except IOError:
        nb_classes = len(all_lab_train)
        x_train_pad = sequence.pad_sequences(list_im_train, value=float(-1), dtype='float32',
                                    padding="post", truncating='post')
        pad_value, _ = x_train_pad[0].shape
        print("pad value = " + str(pad_value))
        x_valid_pad = sequence.pad_sequences(list_im_valid, value=float(-1), dtype='float32',
                                     padding="post", truncating='post', maxlen = pad_value)

        list_labels_train_conv = convert_ch_to_int(list_labels_train, all_lab_train)
        list_labels_valid_conv = convert_ch_to_int(list_labels_valid, all_lab_train)

        y_train_conv_pad = sequence.pad_sequences(list_labels_train_conv, value=float(nb_classes + 1),
                                    dtype='float32', padding="post")
        y_valid_conv_pad = sequence.pad_sequences(list_labels_valid_conv, value=float(nb_classes + 1),
                                     dtype='float32', padding="post")
        pickle.dump([(x_train_pad, y_train_conv_pad), (x_valid_pad, y_valid_conv_pad)],
                    open( 'data/all_data_null_pad_pickle', 'wb' ))
    return (x_train_pad, y_train_conv_pad), (x_valid_pad, y_valid_conv_pad), all_lab_train


def eval_CER_and_WER(list_predictions, list_true_labels):
    """ Compute the Character Error Rate (CER) and Word Error Rate (WER)
    list_predictions and list_true_labels are lists of string, one string is a word (the real word or the
    word that has been predicted by the model)

    The function return the CER and WER measures
    """

    assert(len(list_predictions)==len(list_true_labels))

    cer = 0.
    wer = 0.
    nb_characters = 0
    nb_words = len(list_true_labels)

    print(len(list_predictions))
    for i in range(len(list_predictions)):
        ed, _ = edit_distance(list_predictions[i], list_true_labels[i])

        cer += ed
        if ed != 0:
            wer += 1
        nb_characters += len(list_true_labels[i])

    if nb_characters != 0:
        cer = cer / nb_characters
        print("the Character Error Rate is equal to %.2f" % (cer))
    else:
        print('the number of characters is equal to zero..')
    if nb_words != 0:
        wer = wer / nb_words
        print("the Word Error Rate is equal to %.2f" % (wer))
    else:
        print('the number of words is equal to zero..')

    return cer, wer

def lowest_cost_action(ic, dc, sc, im, dm, sm, cost):
    """Given the following values, choose the action (insertion, deletion,
    or substitution), that results in the lowest cost (ties are broken using
    the 'match' score).  This is used within the dynamic programming algorithm.
    * ic - insertion cost
    * dc - deletion cost
    * sc - substitution cost
    * im - insertion match (score)
    * dm - deletion match (score)
    * sm - substitution match (score)
    """
    best_action = None
    best_match_count = -1
    min_cost = min(ic, dc, sc)
    if min_cost == sc and cost == 0:
        best_action = 'equal'
        best_match_count = sm
    elif min_cost == sc and cost == 1:
        best_action = 'replace'
        best_match_count = sm
    elif min_cost == ic and im > best_match_count:
        best_action = 'insert'
        best_match_count = im
    elif min_cost == dc and dm > best_match_count:
        best_action = 'delete'
        best_match_count = dm
    return best_action

def edit_distance(seq1, seq2, action_function=lowest_cost_action, test=operator.eq):
    """Computes the edit distance between the two given sequences.
    This uses the relatively fast method that only constructs
    two columns of the 2d array for edits.  This function actually uses four columns
    because we track the number of matches too.
    """
    m = len(seq1)
    n = len(seq2)
    # Special, easy cases:
    if np.all(seq1 == seq2):
        return 0, n
    if m == 0:
        return n, 0
    if n == 0:
        return m, 0
    v0 = [0] * (n + 1)     # The two 'error' columns
    v1 = [0] * (n + 1)
    m0 = [0] * (n + 1)     # The two 'match' columns
    m1 = [0] * (n + 1)
    for i in range(1, n + 1):
        v0[i] = i
    for i in range(1, m + 1):
        v1[0] = i
        for j in range(1, n + 1):
            cost = 0 if test(seq1[i - 1], seq2[j - 1]) else 1
            # The costs
            ins_cost = v1[j - 1] + 1
            del_cost = v0[j] + 1
            sub_cost = v0[j - 1] + cost
            # Match counts
            ins_match = m1[j - 1]
            del_match = m0[j]
            sub_match = m0[j - 1] + int(not cost)

            action = action_function(ins_cost, del_cost, sub_cost, ins_match,
                                     del_match, sub_match, cost)

            if action in ['equal', 'replace']:
                v1[j] = sub_cost
                m1[j] = sub_match
            elif action == 'insert':
                v1[j] = ins_cost
                m1[j] = ins_match
            elif action == 'delete':
                v1[j] = del_cost
                m1[j] = del_match
            else:
                raise Exception('Invalid dynamic programming option returned!')
                # Copy the columns over
        for i in range(0, n + 1):
            v0[i] = v1[i]
            m0[i] = m1[i]
    return v1[n], m1[n]



if __name__ == '__main__':

    #list_im, list_labels, all_labels = read_data('./data/', set_to_read='train')


    # Test evaluation
    #list_predictions = ['python', 'helo', 'ward']
    #list_true_labels = ['python', 'hello', 'world']
    #eval_CER_and_WER(list_predictions, list_true_labels)
    load_data_null_pad()
    load_data_white_pad()
