# -*- coding: utf-8 -*-
"""
..author:: Alex Müller, ETH Zürich, Switzerland.
..date:: September 2017

Code for training a LSTM model on peptide sequences followed by sampling novel sequences through the model.
Check the readme for possible flags to use with this script.
"""
import os
import random

import matplotlib
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, LSTM, BatchNormalization, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from progressbar import ProgressBar
from sklearn.model_selection import KFold
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from modlamp.sequences import Random, Helices
from modlamp.core import count_aas

sess = tf.Session()
from keras import backend as K

K.set_session(sess)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

flags = tf.app.flags

flags.DEFINE_string("dataset", "training_sequences_noC.csv", "dataset file (expecting csv)")
flags.DEFINE_string("run_name", "test", "run name for log and checkpoint files")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("epochs", 100, "epochs to train")
flags.DEFINE_integer("layers", 2, "number of layers in the network")
flags.DEFINE_float("valsplit", 0.2, "percentage of the data to use for validation")
flags.DEFINE_integer("neurons", 64, "number of units per layer")
flags.DEFINE_integer("sample", 100, "number of sequences to sample training")
flags.DEFINE_integer("maxlen", 0, "maximum sequence length allowed when sampling new sequences")
flags.DEFINE_float("temp", 2.5, "temperature used for sampling")
flags.DEFINE_string("startchar", "B", "starting character to begin sampling. Default='B' for 'begin'")
flags.DEFINE_float("dropout", 0.2, "dropout to use in every layer; layer 1 gets 1*dropout, layer 2 2*dropout etc.")
flags.DEFINE_bool("train", True, "wether the network should be trained or just sampled from")
flags.DEFINE_float("lr", 0.01, "learning rate to be used with the Adam optimizer")
flags.DEFINE_string("modfile", None, "filename of the pretrained model to used for sampling if train=False")
flags.DEFINE_integer("cv", None, "number of folds to use for cross-validation; if None, no CV is performed")
flags.DEFINE_integer("step", 1, "step size to move window or prediction target")
flags.DEFINE_string("target", "all", "whether to learn all proceeding characters or just the last `one` in sequence")
flags.DEFINE_integer("padlen", 0, "number of spaces to use for padding sequences (if window not 0); if 0, sequences are"
                                  " padded to the length of the longest sequence in the dataset")
flags.DEFINE_integer("window", 0, "window size used to process sequences. If 0, all sequences are padded to the "
                                  "longest sequence length in the dataset")
flags.DEFINE_bool("distance", True, "distance calculation of sampled vs. training sequences in descriptor space")

FLAGS = flags.FLAGS


def _save_flags(flags, filename):
    """Function to save used tf.FLAGS to log-file

    :param flags: tensorflow flags
    :return: saved file
    """
    with open(filename, 'w') as f:
        f.write("Used flags:\n-----------\n")
        for k, v in flags.__dict__['__flags'].items():
            f.write(k + ": " + str(v) + "\n")


def _onehotencode(s, vocab=None):
    """Function to one-hot encode a sring.
    
    :param s: {str} String to encode in one-hot fashion
    :param vocab: vocabulary to use fore encoding, if None, default AAs are used
    :return: one-hot encoded string as a np.array
    """
    if not vocab:
        vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                 'Y', ' ']
    
    # generate translation dictionary for one-hot encoding
    to_one_hot = dict()
    for i, a in enumerate(vocab):
        v = np.zeros(len(vocab))
        v[i] = 1
        to_one_hot[a] = v
    
    result = []
    for l in s:
        result.append(to_one_hot[l])
    result = np.array(result)
    return np.reshape(result, (1, result.shape[0], result.shape[1])), to_one_hot, vocab


def _onehotdecode(matrix, vocab=None, lenmin=1, lenmax=50, filename=None):
    """Decode a given one-hot represented matrix back into sequences

    :param matrix: matrix containing sequence patterns that are one-hot encoded
    :param vocab: vocabulary, if None, standard AAs are used
    :param lenmin: minimum length of sequences to keep
    :param lenmax: maximum length of sequences to keep
    :param filename: filename for saving sequences, if ``None``, sequences are returned in a list
    :return: list of decoded sequences in the range lenmin-lenmax, if ``filename``, they are saved to a file
    """
    if not vocab:
        _, _, vocab = _onehotencode('A')
    if len(matrix.shape) == 2:  # if a matrix containing only one string is supplied
        result = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                aa = np.where(matrix[i, j] == 1.)[0][0]
                result.append(vocab[aa])
        seq = ''.join(result)
        if lenmin <= len(seq) <= lenmax:
            if filename:
                with open(filename, 'wb') as f:
                    f.write(seq)
            else:
                return seq
    
    elif len(matrix.shape) == 3:  # if a matrix containing several strings is supplied
        result = []
        for n in range(matrix.shape[0]):
            oneresult = []
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    aa = np.where(matrix[n, i, j] == 1.)[0][0]
                    oneresult.append(vocab[aa])
            seq = ''.join(oneresult)
            if lenmin <= len(seq) <= lenmax:
                result.append(seq)
        if filename:
            with open(filename, 'wb') as f:
                for s in result:
                    f.write(s + '\n')
        else:
            return result


def _sample_with_temp(preds, temp=1.0):
    """Helper function to sample one letter from a probability array given a temperature.

    :param preds: {np.array} predictions returned by the network
    :param temp: {float} temperature value to sample at.
    """
    streched = np.log(preds) / temp
    vocab_size = len(streched)
    strethced_probs = np.exp(streched) / np.sum(np.exp(streched))
    return np.random.choice(vocab_size, p=strethced_probs)

    
class SequenceHandler(object):
    """
    Class for handling peptide sequences, e.g. loading, one-hot encoding or decoding and saving
    """
    
    def __init__(self, window=0):
        """
        :param window: {str} window used for chopping up sequences. If 0: False
        """
        self.sequences = None
        self.generated = None
        self.X = list()
        self.y = list()
        self.window = window
        # generate translation dictionary for one-hot encoding
        _, self.to_one_hot, self.vocab = _onehotencode('A')
    
    def load_sequences(self, filename):
        """Method to load peptide sequences from a csv file
        
        :param filename: {str} filename of the sequence file to be read (``csv``, one sequence per line)
        :return: sequences in self.sequences
        """
        with open(filename) as f:
            self.sequences = [s.strip() for s in f]
        self.sequences = random.sample(self.sequences, len(self.sequences))  # shuffle sequences randomly
    
    def pad_sequences(self, pad_char=' ', padlen=0):
        """
        Pad all sequences to the longest length (default) or a given length
        
        :param pad_char: {str} Character to pad sequences with
        :param padlen: {int} Custom length of padding to add to all sequences to (optional), default: 0. If
        0, sequences are padded to the length of the longest sequence in the training set. If a window is used and the
        padded sequence is shorter than the window size, it is padded to fit the window.
        """
        if padlen:
            padded_seqs = []
            for seq in self.sequences:
                if len(seq) < self.window:
                    padded_seq = seq + pad_char * (1 + self.window - len(seq))
                else:
                    padded_seq = seq + pad_char * padlen
                padded_seqs.append(padded_seq)
        else:
            length = max([len(seq) for seq in self.sequences])
            padded_seqs = []
            for seq in self.sequences:
                padded_seq = 'B' + seq + pad_char * (length - len(seq))
                padded_seqs.append(padded_seq)
        
        if pad_char not in self.vocab:
            self.vocab += [pad_char]
        
        self.sequences = padded_seqs  # overwrite sequences with padded sequences
    
    def one_hot_encode(self, step=2, target='all'):
        """Chop up loaded sequences into patterns of length ``window`` by moving by stepsize ``step`` and translate
        them with a one-hot vector encoding
        
        :param step: {int} size of the steps to move the window forward
        :param target: {str} whether all proceeding AA should be learned or just the last one in sequence (`all`, `one`)
        :return: one-hot encoded sequence patterns in self.X and corresponding target amino acids in self.y
        """
        if self.window == 0:
            for s in self.sequences:
                self.X.append([self.to_one_hot[char] for char in s[:-step]])
                if target == 'all':
                    self.y.append([self.to_one_hot[char] for char in s[step:]])
                elif target == 'one':
                    self.y.append(s[-step:])
            
            self.X = np.reshape(self.X, (len(self.X), len(self.sequences[0]) - step, len(self.vocab)))
            self.y = np.reshape(self.y, (len(self.y), len(self.sequences[0]) - step, len(self.vocab)))
        
        else:
            for s in self.sequences:
                for i in range(0, len(s) - self.window, step):
                    self.X.append([self.to_one_hot[char] for char in s[i: i + self.window]])
                    if target == 'all':
                        self.y.append([self.to_one_hot[char] for char in s[i + 1: i + self.window + 1]])
                    elif target == 'one':
                        self.y.append(s[-step:])
            
            self.X = np.reshape(self.X, (len(self.X), self.window, len(self.vocab)))
            self.y = np.reshape(self.y, (len(self.y), self.window, len(self.vocab)))

        print("\nData shape:\nX: " + str(self.X.shape) + "\ny: " + str(self.y.shape))

    def analyze_training(self):
        """Method to analyze the distribution of the training data
        
        :return: prints out information about the length distribution of the sequences in ``self.sequences``
        """
        d = GlobalDescriptor(self.sequences)
        d.length()
        print("\nLENGTH DISTRIBUTION OF TRAINING DATA:\n")
        print("Number of sequences:    \t%i" % len(self.sequences))
        print("Mean sequence length:   \t%.1f ± %.1f" % (np.mean(d.descriptor), np.std(d.descriptor)))
        print("Median sequence length: \t%i" % np.median(d.descriptor))
        print("Minimal sequence length:\t%i" % np.min(d.descriptor))
        print("Maximal sequence length:\t%i" % np.max(d.descriptor))
    
    def analyze_generated(self, distances=True):
        """Method to analyze the generated sequences located in `self.generated`.
        
        :param distances: {bool} whether distances in descriptor space should comparing sampled and training
        molecules should be calculated.
        sampled
        :return:
        """
        count = 0
        print("\nNr. of duplicates in generated sequences: %i" % (len(self.generated) - len(set(self.generated))))
        for g in set(self.generated):
            if g in self.sequences:
                count += 1
        print("%.2f percent of generated sequences are present in the training data." % (count / len(self.generated)))

        d = GlobalDescriptor(self.generated)
        d.filter_aa('B')
        d.length()
        print("\nLENGTH DISTRIBUTION OF GENERATED DATA:\n")
        print("Number of valid sequences:\t%i" % len(self.generated))
        print("Mean sequence length:     \t%.1f ± %.1f" % (np.mean(d.descriptor), np.std(d.descriptor)))
        print("Median sequence length:   \t%i" % np.median(d.descriptor))
        print("Minimal sequence length:  \t%i" % np.min(d.descriptor))
        print("Maximal sequence length:  \t%i" % np.max(d.descriptor))

        if distances:
            seq_desc = PeptideDescriptor([s[1:].rstrip() for s in self.sequences], 'PPCALI')
            seq_desc.calculate_autocorr(7)
            gen_desc = PeptideDescriptor(d.sequences, 'PPCALI')
            gen_desc.calculate_autocorr(7)

            # random comparison set
            ran = Random(len(self.generated), np.min(d.descriptor), np.max(d.descriptor))  # generate random seqs
            probas = count_aas(''.join(seq_desc.sequences)).values()  # get the amino acid distribution of training seqs
            ran.generate_sequences(proba=probas)
            ran_desc = PeptideDescriptor(ran.sequences, 'PPCALI')
            ran_desc.calculate_autocorr(7)

            # amphipathic helices comparison set
            hel = Helices(len(self.generated), np.min(d.descriptor), np.max(d.descriptor))
            hel.generate_sequences()
            hel_desc = PeptideDescriptor(hel.sequences, 'PPCALI')
            hel_desc.calculate_autocorr(7)

            # distance calculation
            distance_metrics = ['euclidean']  # , 'cosine']
            for dist in distance_metrics:
                print("\nCalculating distances...")
                desc_dist = distance.cdist(gen_desc.descriptor, seq_desc.descriptor, metric=dist)
                print("\tAverage %s distance in PPCALI descriptor space:\t%.4f" % (dist, np.mean(desc_dist)))
                ran_dist = distance.cdist(ran_desc.descriptor, seq_desc.descriptor, metric=dist)
                print("\tAverage %s distance if randomly sampled seqs:\t%.4f" % (dist, np.mean(ran_dist)))
                hel_dist = distance.cdist(hel_desc.descriptor, seq_desc.descriptor, metric=dist)
                print("\tAverage %s distance if amphipathic helical seqs:\t%.4f" % (dist, np.mean(hel_dist)))

    def save_generated(self, filename):
        """Save all sequences in `self.generated` to file
        
        :param filename: {str} filename to save the sequences to
        :return: saved file
        """
        with open(filename, 'w') as f:
            for s in self.generated:
                f.write(s + '\n')
        

class Model(object):
    """
    Class containing the LSTM model to learn sequential data
    """
    
    def __init__(self, n_vocab, outshape, session_name, n_units=256, batch=64, layers=2, lr=0.001,
                 loss='categorical_crossentropy', dropoutfract=0.1, seed=42):
        """Initialize the model
        
        :param n_vocab: {int} length of vocabulary
        :param outshape: {int} output dimensionality of the model
        :param session_name: {str} custom name for the current session. Will create directory with this name to save
        results / logs to.
        :param n_units: {int} number of LSTM units per layer
        :param batch: {int} batch size
        :param layers: {int} number of layers in the network
        :param loss: {str} applied loss function, choose from available keras loss functions
        :param lr: {float} learning rate to use with Adam optimizer
        :param dropoutfract: {float} fraction of dropout to add to each layer. Layer1 gets 1 * value, Layer2 2 *
        value and so on.
        :param seed {int} random seed used to initialize weights
        """
        random.seed(seed)
        self.seed = seed
        self.weight_init = None
        self.dropout = dropoutfract
        self.inshape = (None, n_vocab)
        self.outshape = outshape
        self.neurons = n_units
        self.layers = layers
        self.losses = list()
        self.val_losses = list()
        self.batchsize = batch
        self.cv_loss = None
        self.cv_loss_std = None
        self.cv_val_loss = None
        self.cv_val_loss_std = None
        self.model = None
        self.losstype = loss
        self.optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.session_name = session_name
        self.logdir = './' + session_name
        if os.path.exists(self.logdir):
            decision = raw_input('\nSession folder already exists!\nDo you want to overwrite the previous session? [y/n] ')
            if decision in ['n', 'no', 'N', 'NO', 'No']:
                self.logdir = './' + raw_input('Enter new session name: ')
                os.makedirs(self.logdir)
        self.checkpointdir = self.logdir + '/checkpoint/'
        if not os.path.exists(self.checkpointdir):
            os.makedirs(self.checkpointdir)
        _, _, self.vocab = _onehotencode('A')
        
        self.initialize_model(self.seed)
    
    def initialize_model(self, seed=42):
        """Method to initialize the model with all parameters saved in the attributes. This method is used during
        initialization of the class, as well as in cross-validation to reinitialize a fresh model for every fold.
        
        :return: initialized model in ``self.model``
        """
        self.weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)  # init weights randomly -0.05 and 0.05
        self.model = Sequential()
        for l in range(self.layers):
            self.model.add(LSTM(units=self.neurons,
                                name='LSTM%i' % (l + 1),
                                input_shape=self.inshape,
                                return_sequences=True,
                                kernel_initializer=self.weight_init,
                                use_bias=True,
                                bias_initializer='zeros',
                                unit_forget_bias=True,
                                dropout=self.dropout * (l + 1)))
        self.model.add(TimeDistributed(Dense(self.outshape, activation='softmax', name='Dense')))
        self.model.compile(loss=self.losstype, optimizer=self.optimizer)
    
    def train(self, x, y, epochs=10, valsplit=0.2, sample=10):
        """Train the model on given training data.
        
        :param x: {array} training data
        :param y: {array} targets for training data in X
        :param epochs: {int} number of epochs to train
        :param valsplit: {float} fraction of data that should be used as validation data during training
        :param sample: {int} number of sequences to sample after every training epoch
        :return: trained model and measured losses in self.model, self.losses and self.val_losses
        """
        writer = tf.summary.FileWriter('./logs/' + self.session_name, graph=sess.graph)
        for e in range(epochs):
            print("Epoch %i" % e)
            checkpoints = [ModelCheckpoint(filepath=self.checkpointdir + 'model_epoch_%i.hdf5' % e, verbose=0)]
            train_history = self.model.fit(x, y, epochs=1, batch_size=self.batchsize, validation_split=valsplit,
                                           shuffle=False, callbacks=checkpoints)
            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=train_history.history['loss'][-1])])
            writer.add_summary(loss_sum, e)
            
            self.losses.append(train_history.history['loss'])
            if valsplit > 0.:
                self.val_losses.append(train_history.history['val_loss'])
                val_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=train_history.history[
                    'val_loss'][-1])])
                writer.add_summary(val_loss_sum, e)
            if sample:
                for s in self.sample(sample):  # sample sequences after every training epoch
                    print(s)
        writer.close()
    
    def cross_val(self, x, y, epochs=10, cv=5, plot=True):
        """Method to perform cross-validation with the model given data X, y
        
        :param x: {array} training data
        :param y: {array} targets for training data in X
        :param epochs: {int} number of epochs to train
        :param cv: {int} fold
        :param plot: {bool} whether the losses should be plotted and saved to the session folder
        :return:
        """
        self.losses = list()  # clean losses if already present
        self.val_losses = list()
        kf = KFold(n_splits=cv)
        cntr = 0
        for train, test in kf.split(x):
            print("\nFold %i" % (cntr + 1))
            self.initialize_model(seed=cntr)  # reinitialize every fold, otherwise it will "remember" previous data
            train_history = self.model.fit(x[train], y[train], epochs=epochs, batch_size=self.batchsize,
                                           validation_data=(x[test], y[test]))
            self.losses.append(train_history.history['loss'])
            self.val_losses.append(train_history.history['val_loss'])
            cntr += 1
        self.cv_loss = np.mean(self.losses, axis=0)
        self.cv_loss_std = np.std(self.losses, axis=0)
        self.cv_val_loss = np.mean(self.val_losses, axis=0)
        self.cv_val_loss_std = np.std(self.val_losses, axis=0)
        if plot:
            self.plot_losses(cv=True)
        print("\n%i-th epoch's %i-fold cross-validation loss: %.4f ± %.4f" %
              (epochs, cv, self.cv_val_loss[-1], self.cv_val_loss_std[-1]))
    
    def plot_losses(self, show=False, cv=False):
        """Plot the losses obtained in training.
        
        :param show: {bool} Whether the plot should be shown or saved. If ``False``, the plot is saved to the
        session folder.
        :param cv: {bool} Whether the losses from cross-validation should be plotted. The standard deviation will be
        depicted as filled areas around the mean curve.
        :return: plot (saved) or shown interactive
        """
        fig, ax = plt.subplots()
        ax.set_title('LSTM Categorical Crossentropy Loss Plot', fontweight='bold', fontsize=16)
        if cv:
            filename = self.logdir + '/' + self.session_name + '_cv_loss_plot.pdf'
            x = range(1, len(self.cv_loss) + 1)
            ax.plot(x, self.cv_loss, '-', color='#FE4365', label='Training')
            ax.plot(x, self.cv_val_loss, '-', color='k', label='Validation')
            ax.fill_between(x, self.cv_loss + self.cv_loss_std, self.cv_loss - self.cv_loss_std,
                            facecolors='#FE4365', alpha=0.5)
            ax.fill_between(x, self.cv_val_loss + self.cv_val_loss_std, self.cv_val_loss - self.cv_val_loss_std,
                            facecolors='k', alpha=0.5)
            ax.set_xlim([0.5, len(self.cv_loss) + 0.5])
            minloss = np.min(self.cv_val_loss)
            plt.text(x=0.5, y=0.5, s='best epoch: ' + str(np.where(minloss == self.cv_val_loss)[0][0]) + ', val_loss: '
                                     + str(minloss.round(4)), transform=ax.transAxes)
        else:
            filename = self.logdir + '/' + self.session_name + '_loss_plot.pdf'
            x = range(1, len(self.losses) + 1)
            ax.plot(x, self.losses, '-', color='#FE4365', label='Training')
            if self.val_losses:
                ax.plot(x, self.val_losses, '-', color='k', label='Validation')
            ax.set_xlim([0.5, len(self.losses) + 0.5])
        ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.legend(loc='best')
        if show:
            plt.show()
        else:
            plt.savefig(filename)
    
    def sample(self, num=10, minlen=7, maxlen=50, start=None, temp=1., show=False):
        """Invoke generation of sequence patterns through sampling from the trained model.
        
        :param num: {int} number of sequences to sample
        :param minlen {int} minimal allowed sequence length
        :param maxlen: {int} maximal length of each pattern generated, if 0, a random length is chosen between 7 and 50
        :param start: {str} start AA to be used for sampling. If ``None``, a random AA is chosen
        :param temp: {float} temperature value to sample at.
        :param show: {bool} whether the sampled sequences should be printed out
        :return: {array} matrix of patterns of shape (num, seqlen, inputshape[0])
        """
        print("\nSampling...\n")
        sampled = []
        pbar = ProgressBar()
        for rs in pbar(range(num)):
            random.seed(rs)
            if not maxlen:  # if the length should be randomly sampled
                longest = np.random.randint(7, 50)
            else:
                longest = maxlen

            if start:
                start_aa = start
            else:  # generate random starting letter
                start_aa = 'B'
            sequence = start_aa  # start with starting letter

            while sequence[-1] != ' ' and len(sequence) <= longest:
                x, _, _ = _onehotencode(sequence)
                preds = self.model.predict(x)[0][0]
                next_aa = _sample_with_temp(preds, temp=temp)
                sequence += self.vocab[next_aa]

            sequence = sequence[len(start_aa):].rstrip()

            if len(sequence) < minlen:
                continue

            sampled.append(sequence)
            if show:
                print(sequence)

        return sampled
    
    def load_model(self, filename):
        """Function to load a trained model from a hdf5 file
        
        :return: model loaded from file in ``self.model``
        """
        self.model.load_weights(filename)


def main(infile, sessname, neurons=256, layers=2, epochs=10, batchsize=64, window=0, step=2, target='all',
         valsplit=0.2, sample=10, aa='B', temperature=0.8, dropout=0.1, train=True, learningrate=0.01, modfile=None,
         samplelength=36, pad=0, dist=False, cv=None):
    
    # loading sequence data, analyze, pad and encode it
    data = SequenceHandler(window=window)
    data.load_sequences(infile)
    data.analyze_training()
    
    # pad sequences
    data.pad_sequences(padlen=pad)

    # one-hot encode padded sequences
    data.one_hot_encode(step=step, target=target)
    
    # building the LSTM model
    model = Model(n_vocab=len(data.vocab), outshape=len(data.vocab), session_name=sessname, n_units=neurons,
                  batch=batchsize, layers=layers, loss='categorical_crossentropy', lr=learningrate,
                  dropoutfract=dropout, seed=42)
    
    if train:
        if cv:
            print("\nPERFORMING %i-FOLD CROSS-VALIDATION...\n" % cv)
            model.cross_val(data.X, data.y, epochs=epochs, cv=cv)
            model.initialize_model()
            model.train(data.X, data.y, epochs=epochs, valsplit=valsplit, sample=0)
        else:
            # training model on data
            print("\nTRAINING MODEL FOR %i EPOCHS...\n" % epochs)
            model.train(data.X, data.y, epochs=epochs, valsplit=valsplit, sample=0)
            model.plot_losses()  # plot loss
    
    else:
        print("\nUSING PRETRAINED MODEL... (%s)\n" % modfile)
        model.load_model(modfile)
    
    # generating new data through sampling
    print("\nSAMPLING %i SEQUENCES...\n" % sample)
    data.generated = model.sample(sample, start=aa, maxlen=samplelength, show=False, temp=temperature)
    data.analyze_generated(distances=dist)
    data.save_generated(model.logdir + '/sampled_sequences_temp' + str(temperature) + '.csv')


if __name__ == "__main__":

    # run main code
    main(infile=FLAGS.dataset, sessname=FLAGS.run_name, batchsize=FLAGS.batch_size, epochs=FLAGS.epochs,
         layers=FLAGS.layers, valsplit=FLAGS.valsplit, neurons=FLAGS.neurons, sample=FLAGS.sample,
         temperature=FLAGS.temp, dropout=FLAGS.dropout, train=FLAGS.train, modfile=FLAGS.modfile,
         learningrate=FLAGS.lr, cv=FLAGS.cv, samplelength=FLAGS.maxlen, window=FLAGS.window,
         step=FLAGS.step, aa=FLAGS.startchar, target=FLAGS.target, pad=FLAGS.padlen, dist=FLAGS.distance)

    # save used flags to log file
    _save_flags(FLAGS, "./" + FLAGS.run_name + "/flags.txt")
