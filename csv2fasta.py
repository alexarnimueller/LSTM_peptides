#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert a .csv file with sequences to a .fasta file
"""

import sys
from modlamp.descriptors import GlobalDescriptor


d = GlobalDescriptor(sys.argv[1])
d.save_fasta(sys.argv[1][:-3] + 'fasta')
