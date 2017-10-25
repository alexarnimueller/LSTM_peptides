#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to perform a grid search with 10-fold cross-validation on different network architectures.
"""

import os

neurons = [24, 32, 48, 64, 128, 256]
layers = [1, 2]
dropout = [0.1, 0.2]

for l in layers:
    for n in neurons:
        for d in dropout:
            print("\nLayers: %i, Neurons: %i, Dropout: %.1f\n" % (l, n, d))
            cmd = "python ~/PycharmProjects/LSTM_peptides/LSTM_peptides.py --window 40 --step 2" + \
                  " --layers %i --neurons %i --dropout %.1f --cv 10 --name cv10_l%i_n%i_d%.1f" % \
                  (l, n, d, l, n, d)
            os.system(cmd)
