# -*- coding: utf-8 -*-
"""
Code for training a LSTM model on peptide sequences followed by generating novel sequences through the model.
Change the filenames in the main() function to adapt it to your case.

..author:: Alex Müller, ETH Zürich, Switzerland.
..date:: June 2017
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Dense, LSTM
from keras.models import Sequential
from progressbar import ProgressBar
from sklearn.model_selection import KFold


class LSTMpeptide(object):
    """
    Class for handling peptide sequences, e.g. one-hot encoding or decoding
    """
    
    def __init__(self):
        """
        :param wdir: {str} working directory
        """
        self.sequences = None
        self.X = list()
        self.y = list()
        self.vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y', ' ']
        # generate translation dictionary for one-hot encoding
        self.to_one_hot = dict()
        for i, a in enumerate(self.vocab):
            v = np.zeros(len(self.vocab))
            v[i] = 1
            self.to_one_hot[a] = v
    
    def load_sequences(self, filename):
        """Method to load peptide sequences from a csv file
        
        :param filename: {str} filename of the sequence file to be read (``csv``, one sequence per line)
        :return: sequences in self.sequences
        """
        with open(filename) as f:
            self.sequences = [s.strip() for s in f]
    
    def one_hot_encode(self, window=22, step=2):
        """Chop up loaded sequences into patterns of length ``window`` by moving by stepsize ``step`` and translate
        them with a one-hot vector encoding
        
        :param window: {int} size of window to move over the sequences in self.sequences
        :param step: {int} size of the steps to move the window forward
        :return: one-hot encoded sequence patterns in self.X and corresponding target amino acids in self.y
        """
        text = ' '.join(self.sequences)
        for i in range(0, len(text) - window, step):
            seq_in = text[i: i + window]  # training data (-> X)
            seq_out = text[i + window]  # target data (-> y)
            self.X.append([self.to_one_hot[char] for char in seq_in])
            self.y.append(self.to_one_hot[seq_out])
        
        self.X = np.reshape(self.X, (len(self.X), window, len(self.vocab)))
        self.y = np.reshape(self.y, (self.X.shape[0], len(self.vocab)))
    
    def decode(self, matrix, lenmin=5, lenmax=50, filename=None):
        """Decode a given one-hot represented matrix back into sequences
        
        :param matrix: matrix containing sequence patterns that are one-hot encoded
        :param lenmin: minimum length of sequences to keep
        :param lenmax: maximum length of sequences to keep
        :param filename: filename for saving sequences, if ``None``, sequences are returned in a list
        :return: list of decoded sequences in the range lenmin-lenmax, if ``filename``, they are saved to a file
        """
        result = []
        for i in range(matrix.shape[0]):  # generate AA by AA
            for j in range(matrix.shape[1]):
                aa = np.where(matrix[i, j] == 1.)[0][0]
                result.append(self.vocab[aa])
        
        sequences = ''.join(result).split(' ')  # join all sequences and split them again at the newline characters
        sequences = [s for s in sequences if len(s) in range(lenmin, lenmax)]
        if filename:
            with open(filename, 'w') as f:
                for s in list(set(sequences)):
                    f.write(s + '\n')
        else:
            return list(set(sequences))  # filter duplicates


class Model(object):
    """
    Class containing the three-layer LSTM model to learn sequential data
    """
    
    def __init__(self, inshape, outshape, n_units=512, loss='categorical_crossentropy', optimizer='adam', seed=1749):
        """Initialize the model
        
        :param inshape: {tuple} input shape to the network, usually X.shape
        :param outshape: {int} output shape of the network (usually size of vocabulary)
        :param n_units: {int} number of LSTM units per layer
        :param loss: {str} applied loss function, choose from available keras loss functions
        :param optimizer: {str} used optimizer, choose from available keras optimizers
        :param seed {int} random seed used to initialize weights
        """
        self.inshape = inshape
        self.outshape = outshape
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)  # init weights randomly between -0.05 and 0.05
        self.model = Sequential()
        self.model.add(LSTM(n_units, name='LSTM1',
                            input_shape=inshape,
                            return_sequences=True,
                            kernel_initializer=weight_init,
                            use_bias=True,
                            bias_initializer='zeros',
                            unit_forget_bias=True))
        self.model.add(LSTM(n_units, name='LSTM2',
                            return_sequences=True,
                            kernel_initializer=weight_init,
                            use_bias=True,
                            bias_initializer='zeros',
                            unit_forget_bias=True))
        self.model.add(LSTM(n_units, name='LSTM3',
                            kernel_initializer=weight_init,
                            use_bias=True,
                            bias_initializer='zeros',
                            unit_forget_bias=True,
                            dropout=0.25))  # final LSTM layer gets 25% dropout
        self.model.add(Dense(outshape, activation='softmax'))  # combining the outputs of the last layer's units
        self.model.compile(loss=loss, optimizer=optimizer)
        
        self.losses = None
        self.val_losses = None
    
    def train(self, X, y, epochs=20, batchsize=128, valsplit=0.1):
        """Train the model on given training data.
        
        :param X: {array} training data
        :param y: {array} targets for training data in X
        :param epochs: {int} number of epochs to train
        :param batchsize: {int} batch size to use for training on X
        :param valsplit: {float} fraction of data that should be used as validation data during training
        :return: trained model and measured losses in self.model, self.losses and self.val_losses
        """
        train_history = self.model.fit(X, y, epochs=epochs, batch_size=batchsize, validation_split=valsplit)
        self.losses = train_history.history['loss']
        self.val_losses = train_history.history['val_loss']
    
    def cross_val(self, X, y, epochs=20, batchsize=128, cv=5):
        """Method to perform crossvalidation with the model given data X, y
        
        :param X: {array} training data
        :param y: {array} targets for training data in X
        :param epochs: {int} number of epochs to train
        :param batchsize: {int} batch size to use for training on X
        :param cv: {int} fold
        :return:
        """
        self.losses = list()  # clean losses if already present
        self.val_losses = list()
        kf = KFold(n_splits=cv)
        cntr = 0
        for train, test in kf.split(X):
            train_history = self.model.fit(X[train], y[train], epochs=epochs, batch_size=batchsize,
                                           validation_data=(X[test], y[test]))
            self.losses.append(train_history.history['loss'])
            self.val_losses.append(train_history.history['val_loss'])
            cntr += 1
        print("%i-fold cross-validation loss: %.4f ± %.4f" % (cv, np.mean(self.val_losses), np.std(self.val_losses)))
    
    def plot_losses(self, filename=None):
        """Plot the losses obtained in training.
        
        :param filename: {str} filename to save the plot to. If ``None``, just show the plot
        :return: plot (saved) or interactive
        """
        fig, ax = plt.subplots()
        ax.set_title('LSTM Categorical Crossentropy Loss Plot', fontweight='bold', fontsize=16)
        ax.plot(range(len(self.losses)), self.losses, '-', color='#FE4365', label='Training')
        ax.plot(range(len(self.val_losses)), self.val_losses, '-', color='k', label='Validation')
        ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([0, len(self.losses)])
        plt.legend(loc='best')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
    
    def generate(self, num, seqlen=22):
        """Invoke generation of sequence patterns.
        
        :param num: {int} number of patterns to generate
        :param seqlen: {int} length of each pattern generated
        :return: {array} matrix of patterns of shape (num, seqlen, inputshape[0])
        """
        X = None
        rs = 0  # initiate random seed, changes for every sequence
        pbar = ProgressBar()
        for s in pbar(range(num)):
            # generate random starting matrix X = random sequence pattern
            np.random.seed = rs
            X = np.zeros((1, self.inshape[0], self.inshape[1]))
            for i in range(self.inshape[0]):
                pos = np.random.randint(0, self.outshape)  # get a random position of the vector
                X[0, i, pos] = 1.  # set this position to 1
            
            # generate sequence patterns
            for i in range(seqlen):  # generate AA by AA
                prediction = self.model.predict(X, verbose=0)  # let the model predict the next most probable AA
                i_max = np.argmax(prediction)  # index of highest probability in prediction
                tmp = np.zeros(
                    self.outshape)  # temporary empty vector to transform prediction to a one-hot representation
                tmp[i_max] = 1.  # AA to one-hot
                # combine the prediction with X and use the new X to predict next AA
                X = np.reshape(np.vstack((X[0, 1:], tmp)), (1, self.inshape[0], self.inshape[1]))
            rs += 1
        return X


def main():
    #### adapt filenames ####
    inputfile = 'training_sequences_filename.csv'
    outputfile = 'filename_to_save_generated_sequences.csv'
    outputfile_plot = 'output_filename_for_plot.pdf'
    #########################
    
    # loading sequence data and encoding it
    data = LSTMpeptide()
    data.load_sequences(inputfile)
    data.one_hot_encode()
    
    # building 3-layer LSTM model
    model = Model((data.X.shape[1], data.X.shape[2]), len(data.vocab), 
                  n_units=512, 
                  loss='categorical_crossentropy', 
                  optimizer='adam', 
                  seed=1749)
    
    # training model on data
    model.train(data.X, data.y,
                epochs=20,
                batchsize=128,
                valsplit=0.1)
    
    # plot loss
    model.plot_losses(outputfile_plot)
    
    # generating new data
    gen = model.generate(3000)  # generation for 3000 cycles
    data.decode(gen, filename=outputfile)  # save generated sequences to a file


if __name__ == "__main__":
    main()
