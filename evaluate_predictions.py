"""
Script to evaluate the predictions done by the CAMP online prediction tool on the LSTM generated sequences
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

pwdir = '/Users/modlab/x/projects/LSTM/'  # working directory
# pwdir = '/Users/arni/polybox/ETH/PhD/documents/publications/LSTM/'  # working directory

rfX = np.empty((1554, 1))
svmX = np.empty((1554, 1))
daX = np.empty((1554, 1))
# read the prediction files
names = ['train', 'prod', 'rand', 'helices']
labels = ['RF', 'SVM', 'DA']
for t in names:
    rf = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_RF_%s.csv' % t, delimiter=',', dtype='object')
    svm = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_SVM_%s.csv' % t, delimiter=',', dtype='object')
    da = np.genfromtxt(pwdir + 'data/predictions/CAMP_Preds_DA_%s.csv' % t, delimiter=',', dtype='object')

    rf_preds = rf[:, 2].astype('float')
    svm_preds = svm[:, 2].astype('float')
    da_preds = da[:, 2].astype('float')

    rfX = np.hstack((rfX, rf_preds.reshape((-1, 1))))
    svmX = np.hstack((svmX, svm_preds.reshape((-1, 1))))
    daX = np.hstack((daX, da_preds.reshape((-1, 1))))

rfX = pd.DataFrame(rfX[:, 1:], columns=names)
svmX = pd.DataFrame(svmX[:, 1:], columns=names)
daX = pd.DataFrame(daX[:, 1:], columns=names)

# calculate p-values with Welch's t-test
for i, data in enumerate([rfX, svmX, daX]):
    print("\nResults for %s:\n========================================" % labels[i])
    for t in names:
        for s in names:
            rslt = ttest_ind(data[t], data[s], equal_var=False)
            print("p-value for %s vs. %s:  \t%.6f" % (t, s, rslt[1]))


# get mean and std
# mean_rf = np.mean(rf_preds)
# mean_svm = np.mean(svm_preds)
# mean_da = np.mean(da_preds)
# std_rf = np.std(rf_preds)
# std_svm = np.std(svm_preds)
# std_da = np.std(da_preds)
#
# means = [mean_rf, mean_svm, mean_da]
# stds = [std_rf, std_svm, std_da]
#
# # get percentage of inactives
# rf_non = len(np.where(rf_preds < 0.5)[0]) / float(len(rf_preds))
# svm_non = len(np.where(svm_preds < 0.5)[0]) / float(len(svm_preds))
# da_non = len(np.where(da_preds < 0.5)[0]) / float(len(da_preds))
# inactives = [rf_non, svm_non, da_non]

# plot the results
dict_font = {'fontsize': 16, 'fontweight': 'bold'}

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
        pc.set_facecolor('#FA6900')
        pc.set_alpha(0.8)
        pc.set_linewidth(1.5)
        pc.set_alpha(0.7)
        pc.set_label(labels[i])
ax.set_xticks([x + 1 for x in range(len(labels))])
ax.set_xticklabels(labels, fontdict=dict_font)
ax.set_ylabel('P(AMP)', fontdict=dict_font)
plt.savefig(pwdir + 'figures/predictions_violin_hel_%s.pdf' % t)


preds_avrg = pd.read_csv('/Users/modlab/x/projects/LSTM/data/predictions/CAMP_allclass_averaged.csv')

colors = ['#542437', '#FA6900', '#77B885', '#69D2E7']
labels = preds_avrg.columns.tolist()
fig, ax = plt.subplots()
for i, d in enumerate(labels):
    vplot = ax.violinplot(preds_avrg[d], positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
    vplot['cbars'].set_edgecolor('black')
    vplot['cmins'].set_edgecolor('black')
    vplot['cmeans'].set_edgecolor('black')
    vplot['cmaxes'].set_edgecolor('black')
    vplot['cmeans'].set_linestyle('--')
    for pc in vplot['bodies']:
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        pc.set_linewidth(1.5)
        pc.set_alpha(0.7)
        pc.set_label(d)
ax.set_xticks([x + 1 for x in range(len(labels))])
ax.set_xticklabels(labels, fontdict=dict_font)
ax.set_ylabel('P(AMP)', fontdict=dict_font)
plt.savefig(pwdir + 'figures/predictions_averaged_violin.pdf')