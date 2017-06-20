"""
Using a trained LSTM to predict novel AMPs
"""

import numpy as np
from keras.models import load_model
from modlamp.core import aa_weights, BaseSequence, save_fasta
from progressbar import ProgressBar


seq_num = 50  # number of sequences to generate, can be smaller because too short sequences are kicked out
seq_len = 15  # approximate length of generated sequences (afterwards joined and split at newlines)
len_range = range(5, 51)  # range of accepted sequence lengths
featlen = 32  # feature length, defined in training
pwdir = '/Users/modlab/x/projects/LSTM/'  # working directory

# # build translation dictionaries
AAs = sorted(aa_weights().keys())
AAs.append(' ')
n_vocab = len(AAs)
aa_to_int = dict((c, i) for i, c in enumerate(AAs))
int_to_aa = dict((i, c) for i, c in enumerate(AAs))

# load the trained model
print('Loading and plotting model...')
model = load_model(pwdir + "data/logs/best_model_15e.hdf5")

# ----------------- sequence generation ------------------
print("Generating novel sequences...\n")
pbar = ProgressBar()
pep = BaseSequence(1)  # storage vehicle
rs = 0
for s in pbar(range(seq_num)):
    np.random.seed = rs
    # generate random starting vector
    pattern = np.random.rand(1, featlen, 1)[0]
    result = []
    for i in range(seq_len):
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result.append(int_to_aa[index])
        pattern = np.append(pattern[1:], index / float(n_vocab))
    pep.sequences.append(''.join(result))
    rs += 1

pep.sequences = ''.join(pep.sequences).split(' ')  # join all sequences and split them at the newline characters
pep.sequences = [s for s in pep.sequences if len(s) in len_range]  # kick out all sequences that are too short
pep.filter_duplicates()  # kick out duplicates

print("Generated sequences: %i \n" % len(pep.sequences))
for s in pep.sequences:
    print(s)
    
save_fasta(pep, pwdir + "data/produced.fasta")  # save the generated sequences

print("\nDone.")
