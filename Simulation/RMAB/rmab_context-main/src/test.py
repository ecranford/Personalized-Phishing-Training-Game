import numpy as np
import pandas as pd
import time
import utils
from itertools import product
from scipy.stats import sem
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import lp_methods
import mdp
import os
import argparse
import tqdm
import itertools
import mdptoolbox
import matplotlib.pyplot as plt
import sys

def get_whittle_indices(T, gamma, N=2):
    '''
    Inputs:
        Ts: Transition matrix of dimensions N X 2 X 2 where axes are:
            start_state, action, end_state
        gamma: Discount factor
        N: num of states
    Returns:
        index: N Tensor of Whittle index for states (0,1,...,N)
    '''
    if N == 2:
        # Matrix equations for state 0
        row1_s0  =   np.stack([np.ones_like(T[0,0,0]) , gamma * T[0,0,0] - 1, gamma * T[0,0,1]    ], -1)
        row2_s0  =   np.stack([np.zeros_like(T[0,1,0]), gamma * T[0,1,0] - 1, gamma * T[0,1,1]    ], -1)
        row3a_s0 =   np.stack([np.ones_like(T[1,0,0]) , gamma * T[1,0,0]    , gamma * T[1,0,1] - 1], -1)
        row3b_s0 =   np.stack([np.zeros_like(T[1,1,0]), gamma * T[1,1,0]    , gamma * T[1,1,1] - 1], -1)

        A1_s0 = np.stack([row1_s0, row2_s0, row3a_s0], -2)
        A2_s0 = np.stack([row1_s0, row2_s0, row3b_s0], -2)
        b_s0 = np.array([0,0,-1], dtype=np.float32)

        # Matrix equations for state 1
        row1_s1  =   np.stack([np.ones_like(T[1,0,0]) , gamma * T[1,0,0]    , gamma * T[1,0,1] - 1], -1)
        row2_s1  =   np.stack([np.zeros_like(T[1,1,0]), gamma * T[1,1,0]    , gamma * T[1,1,1] - 1], -1)
        row3a_s1 =   np.stack([np.ones_like(T[0,0,0]) , gamma * T[0,0,0] - 1, gamma * T[0,0,1]    ], -1)
        row3b_s1 =   np.stack([np.zeros_like(T[0,1,0]), gamma * T[0,1,0] - 1, gamma * T[0,1,1]    ], -1)

        A1_s1 = np.stack([row1_s1, row2_s1, row3a_s1], -2)
        A2_s1 = np.stack([row1_s1, row2_s1, row3b_s1], -2)
        b_s1 = np.array([-1,-1,0], dtype=np.float32)

        # Compute candidate whittle indices
        cnd1_s0 = np.linalg.solve(A1_s0, b_s0)
        cnd2_s0 = np.linalg.solve(A2_s0, b_s0)

        cnd1_s1 = np.linalg.solve(A1_s1, b_s1)
        cnd2_s1 = np.linalg.solve(A2_s1, b_s1)

        ## Following line implements condition checking when candidate1 is correct
        ## It results in an array of size N, with value 1 if candidate1 is correct else 0.
        cand1_s0_mask = (cnd1_s0[0] + 1.0) + gamma * (T[1,0,0] * cnd1_s0[1] + T[1,0,1] * cnd1_s0[2]) >= \
                            1.0 + gamma * (T[1,1,0] * cnd1_s0[1] + T[1,1,1] * cnd1_s0[2])
        cand1_s1_mask = (cnd1_s1[0])       + gamma * (T[0,0,0] * cnd1_s1[1] + T[0,0,1] * cnd1_s1[2]) >= \
                                    gamma * (T[0,1,0] * cnd1_s1[1] + T[0,1,1] * cnd1_s1[2])

        cand2_s0_mask = ~cand1_s0_mask
        cand2_s1_mask = ~cand1_s1_mask

        return np.stack([cnd1_s0[0] * cand1_s0_mask + cnd2_s0[0] * cand2_s0_mask, cnd1_s1[0] * cand1_s1_mask + cnd2_s1[0] * cand2_s1_mask], -1)
    
    elif N == 3:
        # Matrix equations for state 0
        row1_s0  =   np.stack([np.ones_like(T[0,0,0]) , gamma * T[0,0,0] - 1, gamma * T[0,0,1]    , gamma * T[0,0,2]    ], -1)
        row2_s0  =   np.stack([np.zeros_like(T[0,1,0]), gamma * T[0,1,0] - 1, gamma * T[0,1,1]    , gamma * T[0,1,2]    ], -1)
        row3a_s0 =   np.stack([np.ones_like(T[1,0,0]) , gamma * T[1,0,0]    , gamma * T[1,0,1] - 1, gamma * T[1,0,2]    ], -1)
        row3b_s0 =   np.stack([np.zeros_like(T[1,1,0]), gamma * T[1,1,0]    , gamma * T[1,1,1] - 1, gamma * T[1,1,2]    ], -1)
        row4a_s0 =   np.stack([np.ones_like(T[2,0,0]) , gamma * T[2,0,0]    , gamma * T[2,0,1]    , gamma * T[2,0,2] - 1], -1)
        row4b_s0 =   np.stack([np.zeros_like(T[2,1,0]), gamma * T[2,1,0]    , gamma * T[2,1,1]    , gamma * T[2,1,2] - 1], -1)

        A1_s0 = np.stack([row1_s0, row2_s0, row3a_s0, row4a_s0], -2)
        A2_s0 = np.stack([row1_s0, row2_s0, row3a_s0, row4b_s0], -2)
        A3_s0 = np.stack([row1_s0, row2_s0, row3b_s0, row4a_s0], -2)
        A4_s0 = np.stack([row1_s0, row2_s0, row3b_s0, row4b_s0], -2)
        b_s0 = np.array([0,0,-1,-2], dtype=np.float32)

        # Matrix equations for state 1
        row1_s1  =   np.stack([np.ones_like(T[1,0,0]) , gamma * T[1,0,0]    , gamma * T[1,0,1] - 1, gamma * T[1,0,2]    ], -1)
        row2_s1  =   np.stack([np.zeros_like(T[1,1,0]), gamma * T[1,1,0]    , gamma * T[1,1,1] - 1, gamma * T[1,1,2]    ], -1)
        row3a_s1 =   np.stack([np.ones_like(T[2,0,0]) , gamma * T[2,0,0]    , gamma * T[2,0,1]    , gamma * T[2,0,2] - 1], -1)
        row3b_s1 =   np.stack([np.zeros_like(T[2,1,0]), gamma * T[2,1,0]    , gamma * T[2,1,1]    , gamma * T[2,1,2] - 1], -1)
        row4a_s1 =   np.stack([np.ones_like(T[0,0,0]) , gamma * T[0,0,0] - 1, gamma * T[0,0,1]    , gamma * T[0,0,2]    ], -1)
        row4b_s1 =   np.stack([np.zeros_like(T[0,1,0]), gamma * T[0,1,0] - 1, gamma * T[0,1,1]    , gamma * T[0,1,2]    ], -1)

        A1_s1 = np.stack([row1_s1, row2_s1, row3a_s1, row4a_s1], -2)
        A2_s1 = np.stack([row1_s1, row2_s1, row3a_s1, row4b_s1], -2)
        A3_s1 = np.stack([row1_s1, row2_s1, row3b_s1, row4a_s1], -2)
        A4_s1 = np.stack([row1_s1, row2_s1, row3b_s1, row4b_s1], -2)
        b_s1 = np.array([-1,-1,-2,0], dtype=np.float32)

        # Matrix equations for state 2
        row1_s2  =   np.stack([np.ones_like(T[2,0,0]) , gamma * T[2,0,0]    , gamma * T[2,0,1]    , gamma * T[2,0,2] - 1], -1)
        row2_s2  =   np.stack([np.zeros_like(T[2,1,0]), gamma * T[2,1,0]    , gamma * T[2,1,1]    , gamma * T[2,1,2] - 1], -1)
        row3a_s2 =   np.stack([np.ones_like(T[0,0,0]) , gamma * T[0,0,0] - 1, gamma * T[0,0,1]    , gamma * T[0,0,2]    ], -1)
        row3b_s2 =   np.stack([np.zeros_like(T[0,1,0]), gamma * T[0,1,0] - 1, gamma * T[0,1,1]    , gamma * T[0,1,2]    ], -1)
        row4a_s2 =   np.stack([np.ones_like(T[1,0,0]) , gamma * T[1,0,0]    , gamma * T[1,0,1] - 1, gamma * T[1,0,2]    ], -1)
        row4b_s2 =   np.stack([np.zeros_like(T[1,1,0]), gamma * T[1,1,0]    , gamma * T[1,1,1] - 1, gamma * T[1,1,2]    ], -1)

        A1_s2 = np.stack([row1_s2, row2_s2, row3a_s2, row4a_s2], -2)
        A2_s2 = np.stack([row1_s2, row2_s2, row3a_s2, row4b_s2], -2)
        A3_s2 = np.stack([row1_s2, row2_s2, row3b_s2, row4a_s2], -2)
        A4_s2 = np.stack([row1_s2, row2_s2, row3b_s2, row4b_s2], -2)
        b_s2  = np.array([-2,-2,0,-1], dtype=np.float32)

        # Compute candidate whittle indices
        cnd1_s0 = np.linalg.solve(A1_s0, b_s0)
        cnd2_s0 = np.linalg.solve(A2_s0, b_s0)
        cnd3_s0 = np.linalg.solve(A3_s0, b_s0)
        cnd4_s0 = np.linalg.solve(A4_s0, b_s0)

        cnd1_s1 = np.linalg.solve(A1_s1, b_s1)
        cnd2_s1 = np.linalg.solve(A2_s1, b_s1)
        cnd3_s1 = np.linalg.solve(A3_s1, b_s1)
        cnd4_s1 = np.linalg.solve(A4_s1, b_s1)

        cnd1_s2 = np.linalg.solve(A1_s2, b_s2)
        cnd2_s2 = np.linalg.solve(A2_s2, b_s2)
        cnd3_s2 = np.linalg.solve(A3_s2, b_s2)
        cnd4_s2 = np.linalg.solve(A4_s2, b_s2)

        # Following line implements condition checking when candidate1 is correct
        # It results in an array of size N, with value 1 if candidate1 is correct else 0.
        if (cnd1_s0[0] + 1.0) + gamma * (T[1,0,0] * cnd1_s0[1] + T[1,0,1] * cnd1_s0[2] + T[1,0,2] * cnd1_s0[3]) >= \
                            1.0 + gamma * (T[1,1,0] * cnd1_s0[1] + T[1,1,1] * cnd1_s0[2] + T[1,1,2] * cnd1_s0[3]):
            if (cnd1_s0[0] + 2.0) + gamma * (T[2,0,0] * cnd1_s0[1] + T[2,0,1] * cnd1_s0[2] + T[2,0,2] * cnd1_s0[3]) >= \
                            2.0 + gamma * (T[2,1,0] * cnd1_s0[1] + T[2,1,1] * cnd1_s0[2] + T[2,1,2] * cnd1_s0[3]):
                i1 = cnd1_s0[0]
            else:
                i1 = cnd2_s0[0]
        else:
            if (cnd3_s0[0] + 2.0) + gamma * (T[2,0,0] * cnd3_s0[1] + T[2,0,1] * cnd3_s0[2] + T[2,0,2] * cnd3_s0[3]) >= \
                            2.0 + gamma * (T[2,1,0] * cnd3_s0[1] + T[2,1,1] * cnd3_s0[2] + T[2,1,2] * cnd3_s0[3]):
                i1 = cnd3_s0[0]
            else:
                i1 = cnd4_s0[0]
        
        if (cnd1_s1[0] + 2.0) + gamma * (T[2,0,0] * cnd1_s1[1] + T[2,0,1] * cnd1_s1[2] + T[2,0,2] * cnd1_s1[3]) >= \
                        gamma * (T[2,1,0] * cnd1_s1[1] + T[2,1,1] * cnd1_s1[2] + T[2,1,2] * cnd1_s1[3]):
            if (cnd1_s1[0]) + gamma * (T[0,0,0] * cnd1_s1[1] + T[0,0,1] * cnd1_s1[2] + T[0,0,2] * cnd1_s1[3]) >= \
                            gamma * (T[0,1,0] * cnd1_s1[1] + T[0,1,1] * cnd1_s1[2] + T[0,1,2] * cnd1_s1[3]):
                i2 = cnd1_s1[0]
            else:
                i2 = cnd2_s1[0]
        else:
            if (cnd3_s1[0]) + gamma * (T[0,0,0] * cnd3_s1[1] + T[0,0,1] * cnd3_s1[2] + T[0,0,2] * cnd3_s1[3]) >= \
                            gamma * (T[0,1,0] * cnd3_s1[1] + T[0,1,1] * cnd3_s1[2] + T[0,1,2] * cnd3_s1[3]):
                i2 = cnd3_s1[0]
            else:
                i2 = cnd4_s1[0]

        if (cnd1_s2[0]) + gamma * (T[0,0,0] * cnd1_s2[1] + T[0,0,1] * cnd1_s2[2] + T[0,0,2] * cnd1_s2[3]) >= \
                        gamma * (T[0,1,0] * cnd1_s2[1] + T[0,1,1] * cnd1_s2[2] + T[0,1,2] * cnd1_s2[3]):
            if (cnd1_s2[0] + 1.0) + gamma * (T[1,0,0] * cnd1_s2[1] + T[1,0,1] * cnd1_s2[2] + T[1,0,2] * cnd1_s2[3]) >= \
                            1.0 + gamma * (T[1,1,0] * cnd1_s2[1] + T[1,1,1] * cnd1_s2[2] + T[1,1,2] * cnd1_s2[3]):
                i3 = cnd1_s2[0]
            else:
                i3 = cnd2_s2[0]
        else:
            if (cnd3_s2[0] + 1.0) + gamma * (T[1,0,0] * cnd3_s2[1] + T[1,0,1] * cnd3_s2[2] + T[1,0,2] * cnd3_s2[3]) >= \
                            1.0 + gamma * (T[1,1,0] * cnd3_s2[1] + T[1,1,1] * cnd3_s2[2] + T[1,1,2] * cnd3_s2[3]):
                i3 = cnd3_s2[0]
            else:
                i3 = cnd4_s2[0]
            

        # cand2_s0_mask = ~cand1_s0_mask
        # cand2_s1_mask = ~cand1_s1_mask

        # return np.stack([cnd1_s0[0] * cand1_s0_mask + cnd2_s0[0] * cand2_s0_mask, cnd1_s1[0] * cand1_s1_mask + cnd2_s1[0] * cand2_s1_mask], -1)
        # print(cnd1_s0, cnd1_s1, cnd1_s2)
        return np.stack([i1,i2,i3], -1)

GAMMA = 0.99

for i in range(10):
    # T = np.load('../logs/T/' + str(i) + "/T.npy")
    
    T1 = np.array(
        [
            [
                [0.6, 0.4, 0], # L state action 0
                [0.55, 0.45, 0]  # L state action 1
            ],
            [
                [0.8, 0.2, 0], # P state action 0
                [0, 0.2, 0.8]  # P state action 1
            ],
            [
                [0, 0.9, 0.1], # S state action 0
                [0, 0.85, 0.15]  # S state action 1
            ]
        ]
    )

    T2 = np.array(
        [
            [
                [0.6, 0.4, 0], # L state action 0
                [0.55, 0.45, 0]  # L state action 1
            ],
            [
                [0.6, 0.4, 0], # P state action 0
                [0, 0.6, 0.4]  # P state action 1
            ],
            [
                [0, 0.9, 0.1], # S state action 0
                [0, 0.85, 0.15]  # S state action 1
            ]
        ]
    )

    T3 = np.array(
        [
            [
                [0.6, 0.4, 0], # L state action 0
                [0.55, 0.45, 0]  # L state action 1
            ],
            [
                [0.6, 0.4, 0], # P state action 0
                [0, 0.9, 0.1]  # P state action 1
            ],
            [
                [0, 0.9, 0.1], # S state action 0
                [0, 0.85, 0.15]  # S state action 1
            ]
        ]
    )
    
    w = get_whittle_indices(T1, GAMMA, N=3)
    print(T1, w)

    w = get_whittle_indices(T2, GAMMA, N=3)
    print(T2, w)

    w = get_whittle_indices(T3, GAMMA, N=3)
    print(T3, w)
    # if w[0] - w[1] > 1e-3:
    #     print(t, w)
    #     break
    break