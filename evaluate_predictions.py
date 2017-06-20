"""
Script to evaluate the predictions done by the CAMP online prediction tool on the LSTM generated sequences
"""

import numpy as np
import matplotlib.pyplot as plt

pwdir = '/Users/modlab/x/projects/LSTM/'  # working directory

# read the prediction files
ann = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_ANN.csv', delimiter=',', dtype='object')
rf = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_RF.csv', delimiter=',', dtype='object')
svm = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_SVM.csv', delimiter=',', dtype='object')
da = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_DA.csv', delimiter=',', dtype='object')

# average probability (for all but ANN)
rf_preds = rf[:, 2].astype('float')
svm_preds = svm[:, 2].astype('float')
da_preds = da[:, 2].astype('float')

# get mean and std
mean_rf = np.mean(rf_preds)
mean_svm = np.mean(svm_preds)
mean_da = np.mean(da_preds)
std_rf = np.std(rf_preds)
std_svm = np.std(svm_preds)
std_da = np.std(da_preds)

means = [mean_rf, mean_svm, mean_da]
stds = [std_rf, std_svm, std_da]

# plot the results
labels = ['RF', 'SVM', 'DA']
dict_font = {'fontsize': 16, 'fontweight': 'bold'}
c = '#FA6900'
# barplot
# fig, ax = plt.subplots()
# ax.bar(range(len(means)), means, yerr=stds, color=c)
# ax.set_ylabel('P(AMP)', fontdict=dict_font)
# ax.set_xticks(range(len(means)))
# ax.set_xticklabels(labels, fontdict=dict_font)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.savefig(pwdir + 'figures/predictions_barplot.pdf')

# violinplot
fig, ax = plt.subplots()
for i, l in enumerate([rf_preds, svm_preds, da_preds]):
    vplot = ax.violinplot(l, positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
    # crappy adaptions of violin dictionary elements
    vplot['cbars'].set_edgecolor('black')
    vplot['cmins'].set_edgecolor('black')
    vplot['cmeans'].set_edgecolor('black')
    vplot['cmaxes'].set_edgecolor('black')
    vplot['cmeans'].set_linestyle('--')
    for pc in vplot['bodies']:
        pc.set_facecolor(c)
        pc.set_alpha(0.8)
        pc.set_linewidth(1.5)
        pc.set_alpha(0.7)
        pc.set_label(labels[i])
ax.set_xticks([x + 1 for x in range(len(labels))])
ax.set_xticklabels(labels, fontdict=dict_font)
ax.set_ylabel('P(AMP)', fontdict=dict_font)
plt.savefig(pwdir + 'figures/predictions_violin.pdf')
