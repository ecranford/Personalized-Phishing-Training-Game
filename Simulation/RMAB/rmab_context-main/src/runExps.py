#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:20:29 2021

@author: Susobhan Ghosh
"""

import subprocess
import numpy as np
from itertools import product
import sys
import os

idx = int(sys.argv[1])

if idx < 500:
    experiment = 1
    idx = idx-0
elif idx < 1000:
    experiment = 2
    idx = idx-500

# ---------- Experiment 1: Vary n ---------------------

# if experiment==1:
#     L=[100, 1000, 3000]
#     N=[100, 5000, 10000]
#     B=[0.1,0.2,0.3]
#     start_state=[0,1,2]
#     NT=[4,7]
#     TD=[[40,30,20,10],[30,20,10,10,10,10,10]]
#     combinations = list(product(L, N, NT, start_state, B))
#     this_comb = combinations[idx]
#     l,narms,nt,sts,b=this_comb
#     td=TD[NT.index(nt)]
#     td = [(narms//100)*x for x in td]
#     tdstr = " ".join(str(x) for x in td)

if experiment == 1:
    N = [5000]
    L = [100, 200]
    B = [0.1, 0.2]
    EQ = [0]
    QI = [0]
    NK = [5, 10]
    LUD = [2]
    start_state = [-1]
    combinations = list(product(N, L, B, EQ, QI, NK, LUD, start_state))
    this_comb = combinations[idx]
    narms, l, b, eqdist, qinit, nknown, lud, sts = this_comb

# print(f"python3 simulator.py -pc -1 -d rmab_context_features -l 100 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 5000 -lr 1 -N 20 -sts {sts} -b {b}")
# ot = subprocess.run('which python', shell=True)
# ot1 = subprocess.run('which python3', shell=True)
# print(ot.stdout)
# print(ot1.stdout)

    subprocess.run(
        f'python simulator.py -pc -1 -d arpita_healthcare_static -l {l} -s 0 -ws 0 -ls 0 -g 0.99 -S 3 -A 2 -n {narms} -lr 1 -N 2 -sts {sts} -b {b} -lud {lud} -eqdist {eqdist} -qinit {qinit} -nknown {nknown}', shell=True)

if experiment == 2:
    N = [5000]
    L = [100, 200]
    B = [0.3, 0.4, 0.5]
    LUD = [2]
    start_state = [-1]
    combinations = list(product(N, L, B, LUD, start_state))
    this_comb = combinations[idx]
    narms, l, b, lud, sts = this_comb

# print(f"python3 simulator.py -pc -1 -d rmab_context_features -l 100 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 5000 -lr 1 -N 20 -sts {sts} -b {b}")
# ot = subprocess.run('which python', shell=True)
# ot1 = subprocess.run('which python3', shell=True)
# print(ot.stdout)
# print(ot1.stdout)

    subprocess.run(
        f'python simulator.py -pc -1 -d arpita_healthcare_static -l {l} -s 0 -ws 0 -ls 0 -g 0.99 -S 3 -A 2 -n {narms} -lr 1 -N 2 -sts {sts} -b {b} -lud {lud}', shell=True)
