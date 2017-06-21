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

d1 = GlobalDescriptor(pwdir + 'data/generated/final_generated_10e_512_1554.csv')
d2 = GlobalDescriptor(pwdir + 'data/training_sequences_noC.csv')
a = GlobalAnalysis([d1.sequences, d2.sequences], names=['Prediction', 'Training'])

a.plot_summary(pwdir + '/figures/libraries_plot_final.pdf')

# calculate several descriptors
d1.charge_density()
d2.charge_density()
d1.isoelectric_point(append=True)
d2.isoelectric_point(append=True)
d1.aromaticity(append=True)
d2.aromaticity(append=True)
d1.aliphatic_index(append=True)
d2.aliphatic_index(append=True)
d1.instability_index(append=True)
d2.instability_index(append=True)
d1.boman_index(append=True)
d2.boman_index(append=True)
d1.hydrophobic_ratio(append=True)
d2.hydrophobic_ratio(append=True)
d_names = ['Charge Density', 'pI', 'Aromaticity', 'Aliphatic Index', 'Instability Index', 'Boman Index', 'Hydrophobic Ratio']

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

corr_cd = ttest_ind(d1.descriptor[:, 0], d2.descriptor[:, 0], equal_var=False)
corr_pi = ttest_ind(d1.descriptor[:, 1], d2.descriptor[:, 1], equal_var=False)
corr_ar = ttest_ind(d1.descriptor[:, 2], d2.descriptor[:, 2], equal_var=False)
corr_ai = ttest_ind(d1.descriptor[:, 3], d2.descriptor[:, 3], equal_var=False)
corr_ii = ttest_ind(d1.descriptor[:, 4], d2.descriptor[:, 4], equal_var=False)
corr_bi = ttest_ind(d1.descriptor[:, 5], d2.descriptor[:, 5], equal_var=False)
corr_hr = ttest_ind(d1.descriptor[:, 6], d2.descriptor[:, 6], equal_var=False)

statistics = np.array([corr_charge[0], corr_H[0], corr_uH[0], corr_AAc[0], corr_len[0], corr_cd[0], corr_pi[0],
                       corr_ar[0], corr_ai[0], corr_ii[0], corr_bi[0], corr_hr[0]])
pvalues = np.array([corr_charge[1], corr_H[1], corr_uH[1], corr_AAc[1], corr_len[1], corr_cd[1], corr_pi[1],
                    corr_ar[1], corr_ai[1], corr_ii[1], corr_bi[1], corr_hr[1]])

# Kullback-Leibler Divergence
# kl_AA = entropy(a.aafreq[0], a.aafreq[1])
kl_AAc = entropy(aa1.values(), aa2.values())
kl_charge = entropy(a.charge[0], a.charge[1])
kl_H = entropy(a.H[0], a.H[1])
kl_uH = entropy(a.uH[0], a.uH[1])
kl_len = entropy(a.len[0], a.len[1])
kl_cd = entropy(d1.descriptor[:, 0], d2.descriptor[:, 0])
kl_pi = entropy(d1.descriptor[:, 1], d2.descriptor[:, 1])
kl_ar = entropy(d1.descriptor[:, 2], d2.descriptor[:, 2])
kl_ai = entropy(d1.descriptor[:, 3], d2.descriptor[:, 3])
kl_ii = entropy(d1.descriptor[:, 4], d2.descriptor[:, 4])
kl_bi = entropy(d1.descriptor[:, 5], d2.descriptor[:, 5])
kl_hr = entropy(d1.descriptor[:, 6], d2.descriptor[:, 6])


kl = np.array([kl_charge, kl_H, kl_uH, kl_AAc, kl_len, kl_cd, kl_pi, kl_ar, kl_ai, kl_ii, kl_bi, kl_hr])


df = pd.DataFrame(np.vstack((statistics, pvalues, kl)).T, columns=['Statistic', 'p-Value', 'KL Divergence'],
                  index=['Charge', 'Hydrophobicity', 'Hydrophobic Moment', 'AA counts', 'Length'] + d_names)

print(df)
df.to_csv(pwdir + 'data/generated/correlations_all.csv')
