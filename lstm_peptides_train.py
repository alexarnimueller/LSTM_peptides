# -*- coding: utf-8 -*-
"""
Training an LSTM model on antimicrobial peptides
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, LSTM
from keras.models import Sequential
from modlamp.core import aa_weights
from modlamp.descriptors import GlobalDescriptor

starttime = time()

# ----------------- parameters to adapt ------------------
featlen = 22  # feature length (how many AA per step), here 6 * 3.6, representing approx. 6 turns in a helical peptide
step = 2  # AA step size to move for enhancing sequence library
epochs = 2  # number of training epochs
n_units = 128  # number of LSTM units per layer
batch_s = 64  # batch size used for training
pwdir = '/Volumes/Platte1/x/projects/LSTM/'  # working directory
# --------------------------------------------------------


# reading sequence library to learn from
print("Loading training sequences...")
d = GlobalDescriptor('A')
with open(pwdir + 'data/training_sequences_noC.csv') as f:
    d.sequences = [s.strip() for s in f]

# get rid of sequences containing unnatural AAs and calculate length distribution
d.keep_natural_aa()
d.length()
print('\tSequence length distribution: %.1f ± %.1f' % (np.mean(d.descriptor), np.std(d.descriptor)))

print('\ttotal number of sequences: %i' % len(d.sequences))
peptext = ' '.join(d.sequences)  # join the sequences through a space to a global training text

# translation dictionaries int -> AA and vice versa
AAs = sorted(aa_weights().keys())
AAs.append(' ')
n_vocab = len(AAs)
print('\tVocabulary size = nr. of different AAs (including space): %i' % n_vocab)

# generate one-hot vector representation
aa_to_hot = dict()
for i, a in enumerate(AAs):
    v = np.zeros(n_vocab)
    v[i] = 1
    aa_to_hot[a] = v

# cut the text in semi-redundant sequences of featlen AA in a one-hot vector representation
X = []
y = []
for i in range(0, len(peptext) - featlen, step):
    seq_in = peptext[i: i + featlen]  # training data (-> X)
    seq_out = peptext[i + featlen]  # target data (-> y)
    X.append([aa_to_hot[char] for char in seq_in])
    y.append(aa_to_hot[seq_out])
n_patterns = len(X)
print('\tnr. of sequence patterns: %i' % n_patterns)

# reshape X to [samples, steps, features]
X = np.reshape(X, (n_patterns, featlen, n_vocab))
y = np.reshape(y, (n_patterns, n_vocab))
print('\nShape of X: %s' % str(X.shape))
print('Shape of y: %s' % str(y.shape))

# build the deep LSTM model
print('\nBuilding model...')
weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)  # initialize weights randomly between -0.05 and 0.05
model = Sequential()
model.add(LSTM(n_units, input_shape=(X.shape[1], X.shape[2]),
               return_sequences=True,
               kernel_initializer=weight_init,
               use_bias=True,
               bias_initializer='zeros',
               unit_forget_bias=True))
model.add(LSTM(n_units,
               return_sequences=True,
               kernel_initializer=weight_init,
               use_bias=True,
               bias_initializer='zeros',
               unit_forget_bias=True))
model.add(LSTM(n_units,
               kernel_initializer=weight_init,
               use_bias=True,
               bias_initializer='zeros',
               unit_forget_bias=True,
               dropout=0.25))  # final layer gets 25% dropout

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint to save weights to file
check_path = pwdir + "data/logs/best_lstm_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(check_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

# fit the model
print("Training for %i epochs..." % epochs)
train_history = model.fit(X, y, epochs=epochs, batch_size=batch_s, validation_split=0.1, callbacks=[checkpoint])
losses = train_history.history['loss']
val_losses = train_history.history['val_loss']

# plot loss over training epochs
fig, ax = plt.subplots()
ax.set_title('LSTM Categorical Crossentropy Loss Plot', fontweight='bold', fontsize=16)
ax.plot(range(len(losses)), losses, '-', color='#FE4365', label='Training')
ax.plot(range(len(val_losses)), val_losses, '-', color='k', label='Validation')
trn = ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
tst = ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlim([0, len(losses)])
plt.legend(loc='best')
plt.savefig(pwdir + 'figures/loss_plot_%.2f.pdf' % losses[-1])

print("Duration in minutes: %.2f" % ((time() - starttime) / 60.))
