"""
Script to plot and compare the generated sequences by the LSTM to the training data.
"""
import numpy as np
import pandas as pd
from modlamp.analysis import GlobalAnalysis
from modlamp.descriptors import GlobalDescriptor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from modlamp.sequences import Random
from scipy.stats import ttest_ind, entropy

# pwdir = '/Users/arni/polybox/ETH/PhD/documents/publications/LSTM/'  # working directory
pwdir = '/Users/modlab/x/projects/LSTM/'  # working directory

d1 = GlobalDescriptor(pwdir + 'data/training_sequences_noC.csv')
d2 = GlobalDescriptor(pwdir + 'data/generated/final_generated_10e_512_1554.csv')
d3 = GlobalDescriptor(pwdir + 'data/random_1554.fasta')
d4 = GlobalDescriptor(pwdir + 'data/1554_helices.fasta')

d_dummy = GlobalDescriptor(d1.sequences + d2.sequences + d3.sequences + d4.sequences)  # combine for calculations and minmax
# scaling
names = ['Training', 'Prediction', 'Random', 'Helices']
a = GlobalAnalysis([d1.sequences, d2.sequences, d3.sequences, d4.sequences], names=names)

cols = ['#69D2E7', '#FA6900', '#542437', '#77B885']

a.plot_summary(filename=pwdir + '/figures/libraries_with_helix.pdf', colors=cols, plot=False)

# separate boxplot of lengths
fig, ax = plt.subplots()
box = ax.boxplot(a.len, notch=1, vert=1, patch_artist=True)
plt.setp(box['whiskers'], color='black')
plt.setp(box['medians'], linestyle='-', linewidth=1.5, color='black')
for p, patch in enumerate(box['boxes']):
    patch.set(facecolor=cols[p], edgecolor='black', alpha=0.8)
ax.set_ylabel('Sequence Length', fontweight='bold', fontsize=14.)
ax.set_xticks([x + 1 for x in range(len(names))])
ax.set_xticklabels(names, fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.savefig(pwdir + 'figures/length_boxplot.pdf')

# calculate several descriptors
d_dummy.calculate_all()
d_dummy.descriptor = np.hstack((d_dummy.descriptor, np.hstack((a.H[0], a.H[1], a.H[2], a.H[3])).reshape(-1, 1)))
d_dummy.descriptor = np.hstack((d_dummy.descriptor, np.hstack((a.uH[0], a.uH[1], a.uH[2], a.uH[3])).reshape(-1, 1)))

# min max scaling all descriptors
d_dummy.feature_scaling('minmax')

# PCA
pca = PCA(n_components=2)
pca.fit(d_dummy.descriptor)
X = pca.transform(d_dummy.descriptor)
X = X.reshape((4, 1554, 2))
y = np.array([[0] * 1554] + [[1] * 1554] + [[2] * 1554] + [[3] * 1554])

# plot PCA
fig, ax = plt.subplots(figsize=(10, 7))
for i, x in enumerate(X):
    ax.scatter(x[:, 1], x[:, 0], c=cols[i])
ax.set_ylabel('PC1', fontweight='bold', fontsize=16)
ax.set_xlabel('PC2', fontweight='bold', fontsize=16)
plt.savefig(pwdir + 'figures/PCA.pdf')

d1.descriptor = d_dummy.descriptor[:1554]  # split data again
d2.descriptor = d_dummy.descriptor[1554:3108]
d3.descriptor = d_dummy.descriptor[3108:]

# calculate amino acid distribution
daa = GlobalDescriptor(d_dummy.sequences)
daa.count_aa()
daa.feature_scaling('minmax')
d1aa = daa.descriptor[:1554]
d2aa = daa.descriptor[1554:3108]
d3aa = daa.descriptor[3108:]
featnames = daa.featurenames + d_dummy.featurenames + ['Hydrophobicity', 'Hydrophobic Moment']

# Welchs t-test
# statistics = list()
pvalues1 = list()
pvalues2 = list()
for i in range(d1aa.shape[1]):  # loop over AA
    t1 = ttest_ind(d1aa[:, i], d2aa[:, i], equal_var=False)
    # statistics.append(t[0])
    pvalues1.append(t1[1])
    t2 = ttest_ind(d2aa[:, i], d3aa[:, i], equal_var=False)
    pvalues2.append(t2[1])
for i in range(d1.descriptor.shape[1]):  # loop over descriptors
    t1 = ttest_ind(d1.descriptor[:, i], d2.descriptor[:, i], equal_var=False)
    # statistics.append(t[0])
    pvalues1.append(t1[1])
    t2 = ttest_ind(d2.descriptor[:, i], d3.descriptor[:, i], equal_var=False)
    pvalues2.append(t2[1])
    
# Kullback-Leibler Divergence
kl1 = list()
kl2 = list()
for i in range(d1aa.shape[1]):  # loop over AA
    k1 = entropy(d1aa[:, i], d2aa[:, i])
    kl1.append(k1)
    k2 = entropy(d2aa[:, i], d3aa[:, i])
    kl2.append(k2)
for i in range(d1.descriptor.shape[1]):  # loop over descriptors
    k1 = entropy(d1.descriptor[:, i], d2.descriptor[:, i])
    kl1.append(k1)
    k2 = entropy(d2.descriptor[:, i], d3.descriptor[:, i])
    kl2.append(k2)

df = pd.DataFrame(np.vstack((pvalues1, kl1, pvalues2, kl2)).T,
                  columns=['p-value P-T', 'KL divergence P-T', 'p-value P-R', 'KL divergence P-R'], index=featnames)

print(df)

df.to_csv(pwdir + 'data/generated/correlations_all_rand.csv')

# compare random sequences with same AA distribution as training set with training set and generated
# p_train = [0.10893414506010693, 0.0, 0.018943725084211504, 0.025804258475231004, 0.05163942025402515,
#            0.1287740659476498, 0.022404895083284405, 0.07280818319478352, 0.12098643344973578, 0.1455236564788776,
#            0.011465125621928984, 0.026020581600173058, 0.03801106338267561, 0.02030347044099014, 0.03127414320590871,
#            0.05463704069965079, 0.03189220927717173, 0.06440248462560648, 0.013103000710775982, 0.01307209740721283]
#
# r = Random(1554, lenmin=5, lenmax=50)
# r.generate_sequences(proba=p_train)
# r.save_fasta(pwdir + 'data/random_1554_test.fasta')
#
# a = GlobalAnalysis([d1.sequences, d2.sequences, r.sequences], names=['Prediction', 'Training', 'Random'])
# a.plot_summary(pwdir + 'figures/library_compare_rand.pdf')
