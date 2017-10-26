#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to sample with different temperatures from a trained model.
Need to turn off 'ask' when initializing model
"""

import os

sample = 2000
temps = [0.5, 1., 1.5]
name = 'cv10_l2_n128_d02'
sepoch = 97
maxlen = 48

pwd = '/home/arni/pycharmprojects/lstm_peptides/'
modfile = pwd + name + '/checkpoint/model_epoch_%i.hdf5' % sepoch

for t in temps:
    print("\nSampling %i sequences at %.1f temperature..." % (sample, t))
    cmd = "python %sLSTM_peptides.py --train False --modfile %s " \
          "--temp %.1f --sample %i --maxlen %i --name %s" % (pwd, modfile, t, sample, maxlen, name)
    os.system(cmd)
