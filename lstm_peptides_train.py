"""
Training an LSTM model on antimicrobial peptides
"""

from __future__ import print_function

from time import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils
from modlamp.core import aa_weights
from modlamp.descriptors import GlobalDescriptor

starttime = time()

featlen = 32  # feature length
step = 2  # AA step to move for cutting up sequences
epochs = 15  # number of training epochs
batch_s = 128  # batch size used for training
pwdir = '/Volumes/Platte1/x/projects/LSTM/'  # working directory

# reading sequence library to learn from
print("Loading training sequences...")
d = GlobalDescriptor('A')
with open(pwdir + 'data/training_sequences_noC.csv') as f:
    d.sequences = [s.strip() for s in f]

# get rid of sequences containing unnatural AAs
d.keep_natural_aa()
peptides = d.sequences

print('\ttotal number of sequences:', len(peptides))
# peptext = '\n'.join(h.sequences)
peptext = ' '.join(peptides)  # join the sequences through a space to a global training text
print('\ttotal number of AAs:', len(peptext))

# translation dictionaries int -> AA and vice versa
AAs = sorted(aa_weights().keys())
AAs.append(' ')
n_vocab = len(AAs)
print('\tnr. of different AAs (including space):', n_vocab)
aa_to_int = dict((a, i) for i, a in enumerate(AAs))
int_to_aa = dict((i, a) for i, a in enumerate(AAs))

# cut the text in semi-redundant sequences of featlen AA
X = []
y = []
for i in range(0, len(peptext) - featlen, step):
    seq_in = peptext[i: i + featlen]  # training data (-> X)
    seq_out = peptext[i + featlen]  # target data (-> y)
    X.append([aa_to_int[char] for char in seq_in])
    y.append(aa_to_int[seq_out])
n_patterns = len(X)
print('\tnr. of sequence patterns:', n_patterns)

# reshape X to [samples, steps, features]
print('Vectorization...')
X = np.reshape(X, (n_patterns, featlen, 1))
# normalization
X = X / float(n_vocab)
# one hot encoding of the target
y = np_utils.to_categorical(y)

# build the deep LSTM model
print('Building model...')
weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)  # initialize weights randomly between -0.05 and 0.05
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]),
               return_sequences=True,
               kernel_initializer=weight_init,
               use_bias=True,
               bias_initializer='zeros',
               unit_forget_bias=True))
model.add(LSTM(512,
               return_sequences=True,
               kernel_initializer=weight_init,
               use_bias=True,
               bias_initializer='zeros',
               unit_forget_bias=True))
model.add(LSTM(512,
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
plt.savefig(pwdir + 'figures/loss_plot_best.pdf')

print("Duration in minutes: ", (time() - starttime) / 60.)
