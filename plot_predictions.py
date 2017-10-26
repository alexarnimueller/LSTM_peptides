"""
Script to plot violin plots of the predictions obtained from the CAMP online RF prediction tool
"""
import numpy as np
import pandas as pd
from modlamp.plot import plot_violin

f = './desperate/predictions/CAMP_RF_t'
temps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
cols = ['#' + 6 * x for x in ['0', '3', '6', '9', 'a', 'c', 'd', 'f']]
num = 500  # number of examples to randomly choose (as all temps need the same sample number for plotting)

preds = list()
for t in temps:
    d = pd.read_csv(f + str(t) + '.txt', delimiter='\t')
    preds.append(np.random.choice(d['Probability'].tolist(), 1554, replace=False))

plot_violin(preds, colors=cols, bp=True, filename=f + '_plot.pdf')
