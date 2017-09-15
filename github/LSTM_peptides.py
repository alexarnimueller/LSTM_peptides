# -*- coding: utf-8 -*-
"""
..author:: Alex Müller, ETH Zürich, Switzerland.
..date:: September 2017

Code for training a LSTM model on peptide sequences followed by sampling novel sequences through the model.
Change the filenames at the end of the script to adapt it to your needs.
"""
import os
import random

import matplotlib
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, LSTM, BatchNormalization, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from progressbar import ProgressBar
from sklearn.model_selection import KFold

sess = tf.Session()
from keras import backend as K

K.set_session(sess)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

flags = tf.app.flags

flags.DEFINE_string("dataset", "training_sequences_noC.csv", "dataset file (expecting csv)")
flags.DEFINE_string("run_name", "test", "run name for log and checkpoint files")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("epochs", 25, "epochs to train")
flags.DEFINE_integer("layers", 2, "number of layers in the network")
flags.DEFINE_float("valsplit", 0.2, "percentage of the data to use for validation")
flags.DEFINE_integer("neurons", 256, "number of units per layer")
flags.DEFINE_integer("sample", 10, "number of sequences to sample training")
flags.DEFINE_integer("maxlen", 48, "maximum sequence length allowed when sampling new sequences")
flags.DEFINE_float("temp", 0.8, "temperature used for sampling")
flags.DEFINE_string("startchar", "B", "starting character to begin sampling. Default='B' for 'begin'")
flags.DEFINE_float("dropout", 0.1, "dropout to use in every layer; layer 1 gets 1*dropout, layer 2 2*dropout etc.")
flags.DEFINE_bool("train", True, "wether the network should be trained or just sampled from")
flags.DEFINE_float("lr", 0.001, "learning rate to be used with the Adam optimizer")
flags.DEFINE_string("modfile", None, "filename of the pretrained model to used for sampling if train=False")
flags.DEFINE_integer("cv", None, "number of folds to use for cross-validation; if None, no CV is performed")
flags.DEFINE_integer("step", 1, "step size to move window or prediction target")
flags.DEFINE_integer("window", 0, "window size used to process sequences. If 0, all sequences are padded to the "
                                  "longest sequence length in the dataset")

FLAGS = flags.FLAGS


def onehotencode(s, vocab=None):
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


def onehotdecode(matrix, vocab=None, lenmin=1, lenmax=50, filename=None):
    """Decode a given one-hot represented matrix back into sequences

    :param matrix: matrix containing sequence patterns that are one-hot encoded
    :param vocab: vocabulary, if None, standard AAs are used
    :param lenmin: minimum length of sequences to keep
    :param lenmax: maximum length of sequences to keep
    :param filename: filename for saving sequences, if ``None``, sequences are returned in a list
    :return: list of decoded sequences in the range lenmin-lenmax, if ``filename``, they are saved to a file
    """
    if not vocab:
        _, _, vocab = onehotencode('A')
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


class SequenceHandler(object):
    """
    Class for handling peptide sequences, e.g. one-hot encoding or decoding
    """
    
    def __init__(self):
        self.sequences = None
        self.generated = None
        self.X = list()
        self.y = list()
        # generate translation dictionary for one-hot encoding
        _, self.to_one_hot, self.vocab = onehotencode('A')
    
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
        ``False``, sequences are padded to the length of the longest sequence in the training set.
        """
        if not padlen:
            length = max([len(seq) for seq in self.sequences])
            padded_seqs = []
            for seq in self.sequences:
                padded_seq = 'B' + seq + pad_char * (length - len(seq))
                padded_seqs.append(padded_seq)
        
        else:
            padded_seqs = []
            for seq in self.sequences:
                padded_seq = seq + pad_char * padlen
                padded_seqs.append(padded_seq)
        
        if pad_char not in self.vocab:
            self.vocab += [pad_char]
        
        self.sequences = padded_seqs  # overwrite sequences with padded sequences
    
    def one_hot_encode(self, window=0, step=2):
        """Chop up loaded sequences into patterns of length ``window`` by moving by stepsize ``step`` and translate
        them with a one-hot vector encoding
        
        :param window: {int} size of window to move over the sequences in ``self.sequences``. If ``window=0``,
        the whole sequence length is processed. This is needed if the sequences are all padded to the same length.
        :param step: {int} size of the steps to move the window forward
        :return: one-hot encoded sequence patterns in self.X and corresponding target amino acids in self.y
        """
        if window == 0:
            for s in self.sequences:
                self.X.append([self.to_one_hot[char] for char in s[:-step]])
                self.y.append([self.to_one_hot[char] for char in s[step:]])
            
            self.X = np.reshape(self.X, (len(self.X), len(self.sequences[0]) - step, len(self.vocab)))
            self.y = np.reshape(self.y, (len(self.y), len(self.sequences[0]) - step, len(self.vocab)))
        
        else:
            for s in self.sequences:
                for i in range(0, len(s) - window - 1, step):
                    seq_in = s[i: i + window]  # training data (-> X)
                    seq_out = s[i + window]  # target data (-> y)
                    self.X.append([self.to_one_hot[char] for char in seq_in])
                    self.y.append(self.to_one_hot[seq_out])
            
            self.X = np.reshape(self.X, (len(self.X), window, len(self.vocab)))
            self.y = np.reshape(self.y, (self.X.shape[0], 1, self.X.shape[2]))

    def analyze_generated(self):
        """Method to analyze the generated sequences located in `self.generated`.
        
        :return:
        """
        count = 0
        print("Nr. of duplicates in generated sequences: %i" % (len(self.generated) - len(set(self.generated))))
        for g in set(self.generated):
            if g in self.sequences:
                count += 1
        print("%.2f percent of generated sequences are present in the training data." % (count / len(self.generated)))
    
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
                 loss='categorical_crossentropy', dropoutfract=0.1, seed=1749):
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
            decision = input('\nSession folder already exists!\nDo you want to overwrite the previous session? [y/n] ')
            if decision in ['n', 'no', 'N', 'NO', 'No']:
                self.logdir = './' + input('Enter new session name: ')
                os.makedirs(self.logdir)
        self.checkpointdir = self.logdir + '/checkpoint/'
        if not os.path.exists(self.checkpointdir):
            os.makedirs(self.checkpointdir)
        _, _, self.vocab = onehotencode('A')
        
        self.initialize_model(self.seed)
    
    def initialize_model(self, seed=1749):
        """Method to initialize the model with all parameters saved in the attributes. This method is used during
        initialization of the class, as well as in cross-validation to reinitialize a fresh model for every fold.
        
        :return: initialized model in ``self.model``
        """
        self.weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)  # init weights randomly -0.05 and 0.05
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.inshape, name='BatchNorm'))
        for l in range(self.layers):
            self.model.add(LSTM(self.neurons, name='LSTM%i' % (l + 1),
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
    
    def sample(self, num=10, maxlen=50, start=None, temp=1., show=False):
        """Invoke generation of sequence patterns through sampling from the trained model.
        
        :param num: {int} number of sequences to sample
        :param maxlen: {int} maximal length of each pattern generated, if ``seqlen=0``, = longest sequence length
        :param start: {str} start AA to be used for sampling. If ``None``, a random AA is chosen
        :param temp: {float} temperature value to sample at.
        :param show: {bool} whether the sampled sequences should be printed out
        :return: {array} matrix of patterns of shape (num, seqlen, inputshape[0])
        """
        sampled = []
        pbar = ProgressBar()
        if maxlen == 0:
            maxlen = self.inshape
        
        print("\nSampling...\n")
        for rs in pbar(range(num)):
            random.seed = rs
            if start:
                start_aa = start
            else:  # generate random starting letter
                start_aa = 'B'
            sequence = start_aa  # start with starting letter
            
            while sequence[-1] != ' ' and len(sequence) <= maxlen:
                x, _, _ = onehotencode(sequence)
                preds = self.model.predict(x)[0][0]
                next_aa = self.sample_with_temp(preds, temp=temp)
                sequence += self.vocab[next_aa]
            if show:
                print(sequence)
            if start_aa == 'B':
                sampled.append(sequence[1:-1])
            else:
                sampled.append(sequence[:-1])
        return sampled
    
    def sample_with_temp(self, preds, temp=1.):
        """Sample one letter with a given temperature. For that, the softmax results by the network are reverted,
        divided by a temperature and transformed back into probabilities through a binomial distribution.
        
        :param preds: {np.array} predictions returned by the network
        :param temp: {float} temperature value to sample at.
        :return: most probable letter
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def load_model(self, filename):
        """Function to load a trained model from a hdf5 file
        
        :return: model loaded from file in ``self.model``
        """
        self.model.load_weights(filename)


def main(infile, sessname, neurons=256, layers=2, epochs=10, batchsize=64, window=0, step=2, valsplit=0.2, sample=10,
         aa='B', temperature=0.8, dropout=0.1, train=True, learningrate=0.001, modfile=None, samplelength=36, cv=None):
    # loading sequence data and encoding it
    data = SequenceHandler()
    data.load_sequences(infile)
    
    # pad sequences
    if window != 0:
        data.pad_sequences(padlen=int(window / 5))  # padd sequences windows with 1/5 end strings
    else:
        data.pad_sequences(padlen=False)
    
    # one-hot encode padded sequences
    data.one_hot_encode(window=window, step=step)
    print("Data shape: " + str(data.X.shape))
    
    # building the LSTM model
    model = Model(n_vocab=len(data.vocab), outshape=len(data.vocab), session_name=sessname, n_units=neurons,
                  batch=batchsize, layers=layers, loss='categorical_crossentropy', lr=learningrate,
                  dropoutfract=dropout, seed=1749)
    
    if train:
        if cv:
            print("\nPERFORMING %i-FOLD CROSS-VALIDATION...\n" % cv)
            model.cross_val(data.X, data.y, epochs=epochs, cv=cv)
            model.initialize_model()
            model.train(data.X, data.y, epochs=epochs, valsplit=valsplit, sample=0)
        else:
            # training model on data
            print("\nTRAINING MODEL...\n")
            model.train(data.X, data.y, epochs=epochs, valsplit=valsplit, sample=0)
            # plot loss
            model.plot_losses()
    
    else:
        print("\nUSING PRETRAINED MODEL... (%s)\n" % modfile)
        model.load_model(modfile)
    
    # generating new data through sampling
    print("\nSAMPLING %i SEQUENCES...\n" % sample)
    data.generated = model.sample(sample, start=aa, maxlen=samplelength, show=False, temp=temperature)
    data.analyze_generated()
    data.save_generated(model.logdir + '/sampled_sequences.csv')


if __name__ == "__main__":
    main(FLAGS.dataset, sessname=FLAGS.run_name, batchsize=FLAGS.batch_size, epochs=FLAGS.epochs,
         layers=FLAGS.layers, valsplit=FLAGS.valsplit, neurons=FLAGS.neurons, sample=FLAGS.sample,
         temperature=FLAGS.temp, dropout=FLAGS.dropout, train=FLAGS.train, modfile=FLAGS.modfile,
         learningrate=FLAGS.lr, cv=FLAGS.cv, samplelength=FLAGS.maxlen, window=FLAGS.window,
         step=FLAGS.step, aa=FLAGS.startchar)
