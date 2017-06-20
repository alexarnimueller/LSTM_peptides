"""
Script to plot and compare the generated sequences by the LSTM to the training data.
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, entropy
from modlamp.descriptors import GlobalDescriptor
from modlamp.analysis import GlobalAnalysis
from modlamp.core import count_aa


pwdir = '/Users/modlab/x/projects/LSTM/'  # working directory

d1 = GlobalDescriptor(pwdir + 'data/generated/produced_final_15e_1554.csv')
d2 = GlobalDescriptor(pwdir + 'data/training_sequences_noC.csv')
a = GlobalAnalysis([d1.sequences, d2.sequences], names=['Prediction', 'Training'])

a.plot_summary(pwdir + '/figures/libraries_final.pdf')

# get whole histogram of amino acids
s1 = ''.join(d1.sequences)
s2 = ''.join(d2.sequences)
aa1 = count_aa(s1)
aa2 = count_aa(s2)

# Welchs t-test
# corr_AA = ttest_ind(a.aafreq[0], a.aafreq[1], equal_var=False)
corr_AAc = ttest_ind(aa1.values(), aa2.values(), equal_var=False)
corr_charge = ttest_ind(a.charge[0], a.charge[1], equal_var=False)
corr_H = ttest_ind(a.H[0], a.H[1], equal_var=False)
corr_uH = ttest_ind(a.uH[0], a.uH[1], equal_var=False)
corr_len = ttest_ind(a.len[0], a.len[1], equal_var=False)
statistics = np.array([corr_charge[0], corr_H[0], corr_uH[0], corr_AAc[0], corr_len[0]])
pvalues = np.array([corr_charge[1], corr_H[1], corr_uH[1], corr_AAc[1], corr_len[1]])

# Kullback-Leibler Divergence
# kl_AA = entropy(a.aafreq[0], a.aafreq[1])
kl_AAc = entropy(aa1.values(), aa2.values())
kl_charge = entropy(a.charge[0], a.charge[1])
kl_H = entropy(a.H[0], a.H[1])
kl_uH = entropy(a.uH[0], a.uH[1])
kl_len = entropy(a.len[0], a.len[1])
kl = np.array([kl_charge, kl_H, kl_uH, kl_AAc, kl_len])


df = pd.DataFrame(np.vstack((statistics, pvalues, kl)).T, columns=['Statistic', 'p-Value', 'KL Divergence'],
                  index=['Charge', 'Hydrophobicity', 'Hydrophobic Moment', 'AA counts', 'Length'])

print(df)
df.to_csv(pwdir + 'data/generated/correlations.csv')
