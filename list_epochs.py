#!/usr/bin/env python

import os
import sys

pwdir = sys.argv[1]

for f in os.listdir(pwdir):
    if os.path.isdir(f) and f not in [".git", "logs", ".idea"]:
        print(f)
        fdir = os.path.join(pwdir, f)
        for t in os.listdir(fdir):
            if t.endswith("epoch.txt"):
                os.system("cat %s" % os.path.join(fdir, t))
                print("\n")
