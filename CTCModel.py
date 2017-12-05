import keras.backend as K
import tensorflow as tf
import numpy as np
#from misc.decoder import decode_batch, decoding_with_reference, decodingLen_with_reference, fast_decode_batch, \
#    default_decoding
from keras import Input
from keras.engine import Model, Layer
from keras.layers import Lambda, Masking
from keras.models import model_from_json
import pickle
import os
import time
from tensorflow.python.ops import ctc_ops as ctc
from keras.preprocessing import sequence

#from misc.utils_analysis import compute_list_edit_distance, compute_wer_fromLists, tf_set_edit_distance, \
#    tf_edit_distance
#from misc.utils_data import get_sparse_tensor
#from misc.utils_keras import Kreshape_To1D

"""
authors: Yann Soullard, Cyprien Ruffino (2017)
LITIS lab, university of Rouen (France)
- Interim version -

Classe abstraite
Facilite la création de modèles utilisant la loss CTC

Comme Keras ne permet pas de donner des paramètres additionnels à une
fonction de loss, l'astuce est de rajouter une couche Lambda dans le modèle
à entrainer qui va prendre la sortie de la couche dense et calculer la
loss CTC.

On produit deux modèles, le premier aura comme sortie la loss CTC, et sera celui
sur lequel l'entrainement se fera, et le deuxième aura des caractères en sortie et
servira pour la prédiction. On entraine le premier modèle, puis passe ses poids au
deuxième modèle pour la prédiction.

Pour créer un modèle avec cette classe, faire
une classe qui hérite de CTCModel, et construire le
modèle dans __init__, en affectant self.inputs
et self.outputs

Utiliser ensuite compile, fit et predict
comme avec un modèle classique
"""


class CTCModel:

    def __init__(self, inputs, outputs, greedy=True, beam_width=100, top_paths=1, padding=-1, charset=None):
        """
        A override ou réécrire. C'est dans cette fonction qu'il faudra
        affecter self.inputs et self.outputs avec les listes des layers
        d'entrée et de sortie du réseau
        """
        self.model_train = None
        self.model_pred = None
        self.model_eval = None
        self.inputs = inputs
        self.outputs = outputs

        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths
        self.padding = padding
        self.charset = charset


    def compile(self, optimizer):
        """
        A appeler une fois le modèle créé. Compile le modèle en ajoutant
        la loss CTC

        :param optimizer: L'optimizer a utiliser pendant l'apprentissage
        """
        # Calcul du CTC
        labels = Input(name='labels', shape=[None])
        input_length = Input(name='input_length', shape=[1])
        label_length = Input(name='label_length', shape=[1])

        # Lambda layer for computing the loss function
        loss_out = Lambda(self.ctc_loss_lambda_func, output_shape=(1,), name='CTCloss')(
            self.outputs + [labels, input_length, label_length])


        # Lambda layer for the decoding function
        out_decoded_dense = Lambda(self.ctc_complete_decoding_lambda_func, output_shape=(None, None), name='CTCdecode', arguments={'greedy': self.greedy,
                                     'beam_width': self.beam_width, 'top_paths': self.top_paths},dtype="float32")(
            self.outputs + [input_length])


        #Lambda layer for computing the label error rate
        out_analysis = Lambda(self.ctc_complete_analysis_lambda_func, output_shape=(None,), name='CTCanalysis',
                                   arguments={'greedy': self.greedy,
                                              'beam_width': self.beam_width, 'top_paths': self.top_paths},dtype="float32")(
                    self.outputs + [labels, input_length, label_length])


        # create Keras models
        self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
        self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
        self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)
        self.model_eval = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=out_analysis)

        # Compile models
        self.model_train.compile(loss={'CTCloss': lambda yt, yp: yp}, optimizer=optimizer)
        self.model_pred.compile(loss={'CTCdecode': lambda yt, yp: yp}, optimizer=optimizer)
        self.model_eval.compile(loss={'CTCanalysis': lambda yt, yp: yp}, optimizer=optimizer)


    def get_model_train(self):
        """
        :return: Modèle utilisé en interne pour l'entraînement
        """
        return self.model_train

    def get_model_pred(self):
        """
        :return: Modèle utilisé en interne pour la prédiction
        """
        return self.model_pred


    def get_model_eval(self):
        """
        :return: Model used to evaluate a data set
        """
        return self.model_eval


    def fit(self, x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None):
        """

                Permet de lire les données sur un device (CPU) et en
                parallèle de s'entraîner sur un autre device (GPU)

                Les données d'entrée doivent être de la forme :
                  [input_sequences, label_sequences, inputs_lengths, labels_length]

                :param: Paramètres identiques à ceux de keras.engine.Model.fit()
                :return: L'objet History correspondant à l'entrainement
        """

        out = self.model_train.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=callbacks, validation_split=validation_split, validation_data=validation_data,
            shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        self.model_pred.set_weights(self.model_train.get_weights())
        self.model_eval.set_weights(self.model_train.get_weights())
        return out

    def predict(self, x, batch_size=None, verbose=0):
        """ CTC prediction

        Inputs:
            x = Input data as a 3D Tensor (batch_size, max_input_len, dim_features)
            x_len = 1D array with the length of each data in batch_size
            y = Input data as a 2D Tensor (batch_size, max_label_len)
            y_len = 1D array with the length of each labeling
            label_array = list of labels
            pred = return predictions from the ctc (from model_pred)
            eval = return an analysis of ctc prediction (from model_eval)

        Outputs: a list containing:
            out_pred = output of model_pred
            out_eval = output of model_eval
        """

        #model_out = self.model_pred.evaluate(x=x, y=np.zeros(x[0].shape[0]), batch_size=batch_size, verbose=verbose)
        model_out = self.model_pred.predict(x, batch_size=batch_size, verbose=verbose)

        return model_out


    def predict2(self, x, batch_size=None, verbose=0, steps=None):

        """
        The same function as in the Keras Model but with a different function predict_loop for dealing with variable length predictions

        Generates output predictions for the input samples.

                Computation is done in batches.

                # Arguments
                    x: The input data, as a Numpy array
                        (or list of Numpy arrays if the model has multiple outputs).
                    batch_size: Integer. If unspecified, it will default to 32.
                    verbose: Verbosity mode, 0 or 1.
                    steps: Total number of steps (batches of samples)
                        before declaring the prediction round finished.
                        Ignored with the default value of `None`.

                # Returns
                    Numpy array(s) of predictions.

                # Raises
                    ValueError: In case of mismatch between the provided
                        input data and the model's expectations,
                        or in case a stateful model receives a number of samples
                        that is not a multiple of the batch size.
                """
        #[x, x_len] = x
        # Backwards compatibility.
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError('If predicting from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
        # Validate user data.
        x = _standardize_input_data(x, self.model_pred._feed_input_names,
                                    self.model_pred._feed_input_shapes,
                                    check_batch_axis=False)
        if self.model_pred.stateful:
            if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
                raise ValueError('In a stateful network, '
                                 'you should only pass inputs with '
                                 'a number of samples that can be '
                                 'divided by the batch size. Found: ' +
                                 str(x[0].shape[0]) + ' samples. '
                                                      'Batch size: ' + str(batch_size) + '.')

        # Prepare inputs, delegate logic to `_predict_loop`.
        if self.model_pred.uses_learning_phase and not isinstance(K.learning_phase(), int):
            #ins = [x, x_len] + [0.]
            ins = x + [0.]
        else:
            #ins = [x, x_len]
            ins = x
        self.model_pred._make_predict_function()
        f = self.model_pred.predict_function
        out_pred = self._predict_loop(f, ins, batch_size=batch_size,
                                  verbose=verbose, steps=steps)

        list_pred = []
        for elmt in out_pred:
            pred = []
            for val in elmt:
                if val != -1:
                    pred.append(val)
            list_pred.append(pred)

        return list_pred


    @staticmethod
    def ctc_loss_lambda_func(args):
        """
        Function for computing the ctc loss (can be put in a Lambda layer)
        :param args:
            y_pred, labels, input_length, label_length
        :return: CTC loss
        """

        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


    @staticmethod
    def ctc_complete_decoding_lambda_func(args, **arguments):
        """
        Complete CTC decoding using Keras (function K.ctc_decode)
        :param args:
            y_pred, input_length
        :param arguments:
            greedy, beam_width, top_paths
        :return:
            K.ctc_decode with dtype='float32'
        """

        y_pred, input_length = args
        my_params = arguments

        assert (K.backend() == 'tensorflow')

        return K.cast(K.ctc_decode(y_pred, tf.squeeze(input_length), greedy=my_params['greedy'], beam_width=my_params['beam_width'], top_paths=my_params['top_paths'])[0][0], dtype='float32')

    @staticmethod
    def ctc_complete_analysis_lambda_func(args, **arguments):
        """
        Complete CTC analysis using Keras and tensorflow
        WARNING : tf is required
        :param args:
            y_pred, labels, input_length, label_len
        :param arguments:
            greedy, beam_width, top_paths
        :return:
            ler = label error rate
        """

        y_pred, labels, input_length, label_len = args
        my_params = arguments

        assert (K.backend() == 'tensorflow')

        batch = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
        input_length = tf.to_int32(tf.squeeze(input_length))

        greedy = my_params['greedy']
        beam_width = my_params['beam_width']
        top_paths = my_params['top_paths']

        if greedy:
            (decoded, log_prob) = ctc.ctc_greedy_decoder(
                inputs=batch,
                sequence_length=input_length)
        else:
            (decoded, log_prob) = ctc.ctc_beam_search_decoder(
                inputs=batch, sequence_length=input_length,
                beam_width=beam_width, top_paths=top_paths)

        cast_decoded = tf.cast(decoded[0], tf.float32)

        sparse_y = K.ctc_label_dense_to_sparse(labels, tf.cast(tf.squeeze(label_len), tf.int32))
        ed_tensor = tf_edit_distance(cast_decoded, sparse_y, norm=True)
        ler_per_seq = Kreshape_To1D(ed_tensor)

        return K.cast(ler_per_seq, dtype='float32')


    def _predict_loop(self, f, ins, max_len=20, max_value=-1, batch_size=32, verbose=0, steps=None):
        """Abstract method to loop over some data in batches.

        # Arguments
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size.
            verbose: verbosity mode.
            steps: Total number of steps (batches of samples)
                before declaring `_predict_loop` finished.
                Ignored with the default value of `None`.

        # Returns
            Array of predictions (if the model has a single output)
            or list of arrays of predictions
            (if the model has multiple outputs).
        """
        num_samples = self.model_pred._check_num_samples(ins, batch_size,
                                              steps,
                                              'steps')

        if steps is not None:
            # Step-based predictions.
            # Since we do not know how many samples
            # we will see, we cannot pre-allocate
            # the returned Numpy arrays.
            # Instead, we store one array per batch seen
            # and concatenate them upon returning.
            unconcatenated_outs = []
            for step in range(steps):
                batch_outs = f(ins)
                if not isinstance(batch_outs, list):
                    batch_outs = [batch_outs]
                if step == 0:
                    for batch_out in batch_outs:
                        unconcatenated_outs.append([])
                for i, batch_out in enumerate(batch_outs):
                    unconcatenated_outs[i].append(batch_out)

            if len(unconcatenated_outs) == 1:
                return np.concatenate(unconcatenated_outs[0], axis=0)
            return [np.concatenate(unconcatenated_outs[i], axis=0)
                    for i in range(len(unconcatenated_outs))]
        else:
            # Sample-based predictions.
            outs = []
            batches = _make_batches(num_samples, batch_size)
            index_array = np.arange(num_samples)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                if ins and isinstance(ins[-1], float):
                    # Do not slice the training phase flag.
                    ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                else:
                    ins_batch = _slice_arrays(ins, batch_ids)
                batch_outs = f(ins_batch)
                if not isinstance(batch_outs, list):
                    batch_outs = [batch_outs]
                if batch_index == 0:
                    # Pre-allocate the results arrays.
                    for batch_out in batch_outs:
                        #shape = (num_samples, ) + batch_out.shape[1:] # WARNING  10)
                        shape = (num_samples,max_len)
                        outs.append(np.zeros(shape, dtype=batch_out.dtype))
                        #outs.append(np.zeros(shape, dtype=batch_out.dtype))#batch_out.dtype))# WARNING CHANGE FROM THE MAIN CODE
                for i, batch_out in enumerate(batch_outs):
                    #outs[i][batch_start:batch_end] = batch_out # WARNING
                    outs[i][batch_start:batch_end] = sequence.pad_sequences(batch_out, value=float(max_value), maxlen=max_len,
                                     dtype=batch_out.dtype, padding="post")

            if len(outs) == 1:
                return outs[0]
            return outs


    def save_model(self, path_dir, charset=None):
        """ Save a model in path_dir
        save model_train, model_pred and model_eval in json
        save inputs and outputs in json
        save model CTC parameters in a pickle """

        model_json = self.model_train.to_json()
        with open(path_dir + "/model_train.json", "w") as json_file:
            json_file.write(model_json)

        model_json = self.model_pred.to_json()
        with open(path_dir + "/model_pred.json", "w") as json_file:
            json_file.write(model_json)

        model_json = self.model_eval.to_json()
        with open(path_dir + "/model_eval.json", "w") as json_file:
            json_file.write(model_json)

        model_json = self.model_init.to_json()
        with open(path_dir + "/model_init.json", "w") as json_file:
            json_file.write(model_json)

        param = {'greedy': self.greedy, 'beam_width': self.beam_width, 'top_paths': self.top_paths, 'charset': self.charset}

        output = open(path_dir + "/model_param.pkl", 'wb')
        p = pickle.Pickler(output)
        p.dump(param)
        output.close()


    def load_model(self, path_dir, optimizer, initial_epoch=None):
        """ Load a model in path_dir
        load model_train, model_pred and model_eval from json
        load inputs and outputs from json
        load model CTC parameters from a pickle """


        json_file = open(path_dir + '/model_train.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_train = model_from_json(loaded_model_json)

        json_file = open(path_dir + '/model_pred.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_pred = model_from_json(loaded_model_json, custom_objects={"tf": tf})

        json_file = open(path_dir + '/model_eval.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_eval = model_from_json(loaded_model_json, custom_objects={"tf": tf, "ctc": ctc,
                                                                             "tf_edit_distance": tf_edit_distance,
                                                                             "Kreshape_To1D": Kreshape_To1D})

        json_file = open(path_dir + '/model_init.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_init = model_from_json(loaded_model_json, custom_objects={"tf": tf})

        self.inputs = self.model_init.inputs
        self.outputs = self.model_init.outputs

        input = open(path_dir + "/model_param.pkl", 'rb')
        p = pickle.Unpickler(input)
        param = p.load()
        input.close()

        self.greedy = param['greedy'] if 'greedy' in param.keys() else self.greedy
        self.beam_width = param['beam_width'] if 'beam_width' in param.keys() else self.beam_width
        self.top_paths = param['top_paths'] if 'top_paths' in param.keys() else self.top_paths
        self.charset = param['charset'] if 'charset' in param.keys() else self.charset

        self.compile(optimizer)

        if initial_epoch:
            file_weight = path_dir + 'weights.' + "%02d" %(initial_epoch) + '.hdf5'
            print(file_weight)
            if os.path.exists(file_weight):
                self.model_train.load_weights(file_weight)
                self.model_pred.set_weights(self.model_train.get_weights())
                self.model_eval.set_weights(self.model_train.get_weights())
            else:
                print("Weights for epoch ", initial_epoch, " can not be loaded.")
        else:
            print("Training will be start at the beginning.")




def tf_edit_distance(hypothesis, truth, norm=False):
    """ Edit distance using tensorflow

    inputs are tf.Sparse_tensors """

    return tf.edit_distance(hypothesis, truth, normalize=norm, name='edit_distance')


def Kreshape_To1D(my_tensor):
    """ Reshape to a 1D Tensor using K.reshape
    DONT WORK """

    sum_shape = K.sum(K.shape(my_tensor))
    return K.reshape(my_tensor, (sum_shape,))


def get_sparse_tensor(seq_labels, dtype=tf.int32):
    """Create a sparse representation of x.
    Args:
        seq_labels: a list of lists of type dtype where each element is a sequence (e.g. of labels)
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    max_len = 0

    for n, seq in enumerate(seq_labels):
        if len(seq) > max_len:
            max_len = len(seq)
        for i in range(len(seq)):
            indices.append([n, i])
        # indices.extend(zip([n]*len(seq), range(len(seq)))) #
        values.extend(seq)  # every sequences are concatenate in a unique vector of values

    indices = np.asarray(indices, dtype=tf.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(seq_labels), max_len], dtype=tf.int64)  # asarray(indices).max(0)[1]+1], dtype=int64)

    return indices, values, shape


def _standardize_input_data(data, names, shapes=None,
                            check_batch_axis=True,
                            exception_prefix=''):
    """Normalizes inputs and targets provided by users.

    Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.

    # Arguments
        data: User-provided input data (polymorphic).
        names: List of expected array names.
        shapes: Optional list of expected array shapes.
        check_batch_axis: Boolean; whether to check that
            the batch axis of the arrays matches the expected
            value found in `shapes`.
        exception_prefix: String prefix used for exception formatting.

    # Returns
        List of standardized input arrays (one array per model input).

    # Raises
        ValueError: in case of improperly formatted user-provided data.
    """
    if not names:
        if data is not None and hasattr(data, '__len__') and len(data):
            raise ValueError('Error when checking model ' +
                             exception_prefix + ': '
                             'expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]
    if isinstance(data, dict):
        arrays = []
        for name in names:
            if name not in data:
                raise ValueError('No data provided for "' +
                                 name + '". Need data for each key in: ' +
                                 str(names))
            arrays.append(data[name])
    elif isinstance(data, list):
        if len(data) != len(names):
            if data and hasattr(data[0], 'shape'):
                raise ValueError('Error when checking model ' +
                                 exception_prefix +
                                 ': the list of Numpy arrays '
                                 'that you are passing to your model '
                                 'is not the size the model expected. '
                                 'Expected to see ' + str(len(names)) +
                                 ' array(s), but instead got '
                                 'the following list of ' + str(len(data)) +
                                 ' arrays: ' + str(data)[:200] +
                                 '...')
            else:
                if len(names) == 1:
                    data = [np.asarray(data)]
                else:
                    raise ValueError(
                        'Error when checking model ' +
                        exception_prefix +
                        ': you are passing a list as '
                        'input to your model, '
                        'but the model expects '
                        'a list of ' + str(len(names)) +
                        ' Numpy arrays instead. '
                        'The list you passed was: ' +
                        str(data)[:200])
        arrays = data
    else:
        if not hasattr(data, 'shape'):
            raise TypeError('Error when checking model ' +
                            exception_prefix +
                            ': data should be a Numpy array, '
                            'or list/dict of Numpy arrays. '
                            'Found: ' + str(data)[:200] + '...')
        if len(names) > 1:
            # Case: model expects multiple inputs but only received
            # a single Numpy array.
            raise ValueError('The model expects ' + str(len(names)) + ' ' +
                             exception_prefix +
                             ' arrays, but only received one array. '
                             'Found: array with shape ' + str(data.shape))
        arrays = [data]

    # Make arrays at least 2D.
    for i in range(len(names)):
        array = arrays[i]
        if len(array.shape) == 1:
            array = np.expand_dims(array, 1)
            arrays[i] = array

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is None:
                continue
            array = arrays[i]
            if len(array.shape) != len(shapes[i]):
                raise ValueError('Error when checking ' + exception_prefix +
                                 ': expected ' + names[i] +
                                 ' to have ' + str(len(shapes[i])) +
                                 ' dimensions, but got array with shape ' +
                                 str(array.shape))
            for j, (dim, ref_dim) in enumerate(zip(array.shape, shapes[i])):
                if not j and not check_batch_axis:
                    # skip the first axis
                    continue
                if ref_dim:
                    if ref_dim != dim:
                        raise ValueError(
                            'Error when checking ' + exception_prefix +
                            ': expected ' + names[i] +
                            ' to have shape ' + str(shapes[i]) +
                            ' but got array with shape ' +
                            str(array.shape))
    return arrays


def _slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `_slice_arrays(x, indices)`

    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    # Returns
        A slice of the array(s).
    """
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def _make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    """
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]
