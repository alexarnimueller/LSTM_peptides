#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to sample with different temperatures from a trained model.
Need to turn off 'ask' when initializing model
"""

import os

sample = 1000
temps = [.5]
name = 'desperate'
sepoch = 249
l = 1
n = 64
d = 0.2

modfile = name + '/checkpoint/model_epoch_%i.hdf5' % sepoch

for t in temps:
    print("\nSampling %i sequences at %.1f temperature..." % (sample, t))
    cmd = "python ~/PycharmProjects/LSTM_peptides/LSTM_peptides.py --train False --modfile %s --layers %i " \
          "--neurons %i --dropout %.1f --temp %.1f --sample %i --name %s" % \
          (modfile, l, n, d, t, sample, name)
    os.system(cmd)
