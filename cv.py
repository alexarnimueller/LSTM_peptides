#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to perform a grid search with 10-fold cross-validation on different network architectures.
"""

import os

neurons = [24, 32, 48, 64, 128, 256, 512]
layers = [1, 2]
dropout = [0.1, 0.2]

for l in layers:
    for n in neurons:
        for d in dropout:
            print("\nLayers: %i, Neurons: %i, Dropout: %.1f\n" % (l, n, d))
            cmd = "python ./LSTM_peptides.py --layers %i --neurons %i --dropout %.1f --cv 5 --epochs 200 " \
                  "--name cv5_l%i_n%i_d%s" % (l, n, d, l, n, str(d)[-1])
            os.system(cmd)
