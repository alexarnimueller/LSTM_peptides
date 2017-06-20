"""
Using a trained LSTM to predict novel AMPs
"""

import numpy as np
from keras.models import load_model
from modlamp.core import aa_weights, BaseSequence, save_fasta
from progressbar import ProgressBar

# ----------------- parameters to adapt ------------------
seq_num = 10  # number of sequences to generate, can be smaller because too short sequences are kicked out
seq_len = 15  # approximate length of generated sequences (afterwards joined and split at newlines)
len_range = range(5, 51)  # range of accepted sequence lengths
featlen = 22  # feature length, meaning how many amino acids are presented to the network at one step
pwdir = '/Volumes/Platte1/x/projects/LSTM/'  # working directory
# --------------------------------------------------------


# get AA vocabulary for translation
AAs = sorted(aa_weights().keys())
AAs.append(' ')
n_vocab = len(AAs)

# load the trained model
print('Loading trained model...')
model = load_model(pwdir + "data/logs/best_lstm_01-2.6122.hdf5")

# ----------------- sequence generation ------------------
print("Generating novel sequences...\n")
pbar = ProgressBar()
pep = BaseSequence(1)  # storage vehicle
rs = 0  # initiate random seed, changes for every sequence
for s in pbar(range(seq_num)):
    # generate random starting matrix X
    np.random.seed = rs
    X = np.zeros((1, featlen, n_vocab))
    for i in range(featlen):
        rand_aa = np.random.randint(0, n_vocab)  # get a random position of the vector
        X[0, i, rand_aa] = 1.  # set this position to 1
    
    # generate sequence
    result = []
    for i in range(seq_len):  # generate AA by AA
        prediction = model.predict(X, verbose=0)  # let the model predict the next most probable AA
        i_max = np.argmax(prediction)  # index of highest probability in prediction
        result.append(AAs[i_max])  # translate prediction back to the corresponding AA
        tmp = np.zeros(n_vocab)  # temporary empty vector to transform prediction to a one-hot representation
        tmp[i_max] = 1.  # AA to one-hot
        # combine the prediction with X and use the new X to predict next AA
        X = np.reshape(np.vstack((X[0, 1:], tmp)), (1, featlen, n_vocab))
    
    pep.sequences.append(''.join(result))  # join the AAs to a peptide sequence
    rs += 1  # change the random seed for invoking the next sequence

pep.sequences = ''.join(pep.sequences).split(' ')  # join all sequences and split them at the newline characters
pep.sequences = [s for s in pep.sequences if len(s) in len_range]  # kick out all sequences that are too short
pep.filter_duplicates()  # kick out duplicates

print("Generated sequences: %i \n" % len(pep.sequences))
for s in pep.sequences:
    print(s)
    
save_fasta(pep, pwdir + "data/test.fasta")  # save the generated sequences

print("\nDone.")
