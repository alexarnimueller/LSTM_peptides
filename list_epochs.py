#!/usr/bin/env python

import os
import sys

pwdir = sys.argv[1]

print("Name,Epoch,Loss")

for f in os.listdir(pwdir):
    if os.path.isdir(f) and f not in [".git", "logs", ".idea"]:
        fdir = os.path.join(pwdir, f)
        for t in os.listdir(fdir):
            if t.endswith("best_epoch.txt"):
                with open(os.path.join(fdir, t), 'r') as openf:
                    next(openf)
                    next(openf)
                    reslt = list()
                    for l in openf:
                        reslt.append(l.strip().split("\t")[-1])
                    print(f + ',' + ','.join(reslt))
