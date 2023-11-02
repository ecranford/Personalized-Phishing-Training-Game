from gurobipy import *
import numpy as np
import sys
import time
import itertools
import random


def make_T_from_q(Q):

    T = np.zeros(Q.shape)

    T[:, :, -1] = Q[:, :, -1]
    for i in list(range(Q.shape[2]-1))[::-1]:
        T[:, :, i] = Q[:, :, i] - Q[:, :, i+1]

    return T


# Check that a T respects:
# - Q is increasing in s wrt to a and k
# - Q is increasing in a wrt to s and k
def check_T_strict(T):
    S = T.shape[0]
    A = T.shape[1]
    Q = np.zeros((S, A, S))

    for s in range(S):
        for a in range(A):
            for k in range(S):
                Q[s, a, k] = T[s, a, k:].sum()

    # Covers the p11 > p01
    for k in range(S):
        for a in range(A):

            non_decreasing_in_S = True
            previous_value = 0
            for s in range(S):
                non_decreasing_in_S = Q[s, a, k] >= previous_value
                if not non_decreasing_in_S:
                    return False
                previous_value = Q[s, a, k]

    # Ensure that action effects
    # does this check preclude the first? No.
    # I think this covers p11a > p11p but need to verify
    for s in range(S):
        for k in range(S):

            non_decreasing_in_a = True
            previous_value = 0
            for a in range(A):
                non_decreasing_in_a = Q[s, a, k] >= previous_value
                if not non_decreasing_in_a:
                    return False
                previous_value = Q[s, a, k]

    return True


# Check that a T respects:
# - Q is increasing in s wrt to a and k
def check_T_puterman(T):
    S = T.shape[0]
    A = T.shape[1]
    Q = np.zeros((S, A, S))

    for s in range(S):
        for a in range(A):
            for k in range(S):
                Q[s, a, k] = T[s, a, k:].sum()

    # Covers the p11 > p01
    for k in range(S):
        for a in range(A):

            non_decreasing_in_S = True
            previous_value = 0
            for s in range(S):
                non_decreasing_in_S = Q[s, a, k] >= previous_value
                if not non_decreasing_in_S:
                    return False
                previous_value = Q[s, a, k]

    return True


def no_check(T):
    return True


def random_T(S, A, check_function=check_T_strict):

    T = None
    T_passed_check = False
    count_check_failures = 0
    while (not T_passed_check):
        count_check_failures += 1
        if count_check_failures % 1000 == 0:
            print('count_check_failures:', count_check_failures)
        T = np.random.dirichlet(np.ones(S), size=(S, A))
        T_passed_check = check_function(T)

    return T


def get_full_random_experiment(N, S, A, REWARD_BOUND):

    T = np.zeros((N, S, A, S))
    for i in range(N):
        T[i] = random_T(S, A, check_function=no_check)

    R = np.sort(np.random.rand(N, S), axis=1)*REWARD_BOUND
    # R = np.array([np.arange(S) for _ in range(N)])

    C = np.concatenate([[0], np.sort(np.random.rand(A-1))])

    B = N/2*A

    return T, R, C, B


def jitter(T):
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            noise = np.random.rand(T.shape[2])*0.05
            T[i, j] = T[i, j]+noise
            T[i, j] = T[i, j] / T[i, j].sum()
    return T


def easy_3x3(N):

    T1 = np.array([[[0.7, 0.2, 0.10],  # for state 0 action 0
                    [0.50, 0.40, 0.10],  # for state 0 action 1
                    [0.10, 0.70, 0.20]],  # for state 0 action 2

                   [[0.40, 0.40, 0.20],  # for state 1 action 0
                    [0.30, 0.60, 0.10],  # for state 1 action 1
                    [0.00, 0.10, 0.90]],  # for state 1 action 2

                   [[0.20, 0.40, 0.40],  # for state 2 action 0
                    [0.10, 0.20, 0.70],  # for state 2 action 1
                    [0.00, 0.01, 0.99]],  # for state 2 action 2

                   ])

    T = np.array([jitter(T1) for _ in range(N)])

    R = np.array([[0, 1, 2] for _ in range(N)])  # rewards
    C = np.array([0, 1, 2])
    B = N/2

    start_state = np.ones(N)*0

    return T, R, C, B, start_state


def verify_T_matrix(T):

    valid = True
    # print(T[0, 0, 1], T[0, 1, 1])
    valid &= T[0, 0, 1] <= T[0, 1, 1]  # non-oscillate condition
    # print(valid)
    valid &= T[1, 0, 1] <= T[1, 1, 1]  # must be true for active as well
    # print(valid)
    # action has positive "maintenance" value
    valid &= T[0, 1, 1] <= T[1, 1, 1]
    # print(valid)
    # action has non-negative "influence" value
    valid &= T[0, 0, 1] <= T[1, 0, 1]
    # print(valid)
    return valid


def epsilon_clip(T, epsilon):
    return np.clip(T, epsilon, 1-epsilon)


def smooth_real_probs(T, epsilon):
    if T[1, 1] < T[0, 1]:

        # make p11 and p01 equal so we can properly simulate
        # action effects

        # If it looks like this, make t01 = t11
        # [[0.0,  1.0],
        #  [0.01, 0.99]]]

        # If it looks like this, make t11 = t01
        # [[0.95, 0.05],
        #  [1.0,  0.0]]]

        if T[0, 1] >= 0.5:
            T[0, 1] = T[1, 1]
        else:
            T[1, 1] = T[0, 1]

        T[0, 0] = 1 - T[0, 1]
        T[1, 0] = 1 - T[1, 1]

    return T

def circulant_dynamics_prewards(N, b=0.2):
    # WIQL paper: circulant dynamics

    # the T probs are given in the paper
    P1 = [[0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5],
            [0.5, 0, 0, 0.5]]
    
    P0 = [[0.5, 0, 0, 0.5],
            [0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5]]

    T1 = np.array([[P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], [P0[3], P1[3]]])
    T = np.array([T1 for _ in range(N)])
    F = np.array([0.5 for _ in range(N)])

    R1 = np.array([0, 1, 1, 2])
    R = np.array([R1 for _ in range(N)])

    C = np.array([0, 1])

    return T, R, C, F

def circulant_dynamics(N, b=0.2):
    # WIQL paper: circulant dynamics

    # the T probs are given in the paper
    P1 = [[0.5, 0.5, 0, 0],
          [0, 0.5, 0.5, 0],
          [0, 0, 0.5, 0.5],
          [0.5, 0, 0, 0.5]]
    
    P0 = [[0.5, 0, 0, 0.5],
          [0.5, 0.5, 0, 0],
          [0, 0.5, 0.5, 0],
          [0, 0, 0.5, 0.5]]

    T1 = np.array([[P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], [P0[3], P1[3]]])
    T = np.array([T1 for _ in range(N)])
    F = np.array([0.5 for _ in range(N)])

    R1 = np.array([-1, 0, 0, 1])
    R = np.array([R1 for _ in range(N)])

    C = np.array([0, 1])

    return T, R, C, F
    
def behavorial(N):

    T1 = np.array(
        [
            [
                [0.705976096, 1-0.705976096], # G state action 0
                [0.845528455, 1-0.845528455]  # G state action 1
            ],
            [
                [0.554051323, 1-0.554051323], # B state action 0
                [0.738453687, 1-0.738453687]  # B state action 1
            ],
        ]
    )

    T2 = np.array(
        [
            [
                [0.849293564, 1-0.849293564],  # G state action 0
                [0.906849315, 1-0.906849315] # G state action 1
            ],
            [
                [0.754189944, 1-0.754189944],  # B state action 0
                [0.868103212, 1-0.868103212] # B state action 1
            ],
        ]
    )

    T3 = np.array(
        [
            [
                [0.758725341, 1-0.758725341],  # G state action 0
                [0.888413852, 1-0.888413852] # G state action 1
            ],
            [
                [0.607401606, 1-0.607401606],  # B state action 0
                [0.820343599, 1-0.820343599] # B state action 1
            ],
        ]
    )

    T4 = np.array(
        [
            [
                [0.828947368, 1-0.828947368],  # G state action 0
                [0.879672639, 1-0.879672639] # G state action 1
            ],
            [
                [0.768219178, 1-0.768219178],  # B state action 0
                [0.833821706, 1-0.833821706] # B state action 1
            ],
        ]
    )
    
    r = [1, 1/2, 0]

    
    T = []
    R = []
    F = []

    for _ in range(1):
        T.append(T1)
        R.append(r)
        F.append([1, 0, 0, 0])

    for _ in range(1):
        T.append(T2)
        R.append(r)
        F.append([0, 1, 0, 0])

    for _ in range(1):
        T.append(T3)
        R.append(r)
        F.append([0, 0, 1, 0])

    for _ in range(1):
        T.append(T4)
        R.append(r)
        F.append([0, 0, 0, 1])
        
    T = np.array(T)
    R = np.array(R)
    C = np.array([0, 1])
    F = np.array(F)
    
    file1 = open("data/types.txt", "w")
    file2 = open("data/rewards.txt", "w")
    file3 = open("data/costs.txt", "w")
    file4 = open("data/f.txt", "w")
    
    file1.write(str(T))
    file2.write(str(R))
    file3.write(str(C))
    file4.write(str(F))
    
    return T, R, C, F


def simple_spam_ham(N):
    #Drew: here is where you update the transition probabilities for each type
    T1 = np.array(
        [
            [
                [0.464285714, 0.379343629, 0.156370656], # G state action 0
                [0.608870968, 0.326612903, 0.064516129]  # G state action 1
            ],
            [
                [0.2848, 0.4072, 0.308], # M state action 0
                [0.475862069, 0.403448276, 0.120689655]  # M state action 1
            ],
            [
                [0.119891008, 0.295186194, 0.584922797], # B state action 0
                [0.225454545, 0.458181818, 0.316363636]  # B state action 1
            ],
        ]
    )

    T2 = np.array(
        [
            [
                [0.722785368, 0.243278978, 0.033935654], # G state action 0
                [0.763205829, 0.220400729, 0.016393443]  # G state action 1
            ],
            [
                [0.560043668, 0.339519651, 0.100436681], # M state action 0
                [0.641434263, 0.322709163, 0.035856574]  # M state action 1
            ],
            [
                [0.314893617, 0.382978723, 0.30212766], # B state action 0
                [0.45, 0.316666667, 0.233333333]  # B state action 1
            ],
        ]
    )

    T3 = np.array(
        [
            [
                [0.610271903, 0.308660624, 0.081067472], # G state action 0
                [0.75, 0.213675214, 0.036324786]  # G state action 1
            ],
            [
                [0.344251766, 0.384714194, 0.27103404], # M state action 0
                [0.455470738, 0.389312977, 0.155216285]  # M state action 1
            ],
            [
                [0.1, 0.278676471, 0.621323529], # B state action 0
                [0.217261905, 0.449404762, 0.333333333]  # B state action 1
            ],
        ]
    )

    T4 = np.array(
        [
            [
                [0.676223776, 0.275174825, 0.048601399], # G state action 0
                [0.775312067, 0.20665742, 0.018030513]  # G state action 1
            ],
            [
                [0.585325639, 0.346248969, 0.068425392], # M state action 0
                [0.693877551, 0.262390671, 0.043731778]  # M state action 1
            ],
            [
                [0.456221198, 0.400921659, 0.142857143], # B state action 0
                [0.53030303, 0.393939394, 0.075757576]  # B state action 1
            ],
        ]
    )
    
    #Drew: here is where you determine the reward for each state, if there are two states you can use r = [1, 0]
    r = [1, 1/2, 0]

    
    T = []
    R = []
    F = []

    #Drew: here is where you update the number of players in each type by just adjusting the value in the range which is percentage in that type * N (total players)
    for _ in range(1):
        T.append(T1)
        R.append(r)
        F.append([1, 0, 0, 0])

    for _ in range(1):
        T.append(T2)
        R.append(r)
        F.append([0, 1, 0, 0])

    for _ in range(1):
        T.append(T3)
        R.append(r)
        F.append([0, 0, 1, 0])

    for _ in range(1):
        T.append(T4)
        R.append(r)
        F.append([0, 0, 0, 1])
        
    T = np.array(T)
    R = np.array(R)
    C = np.array([0, 1])
    F = np.array(F)
    
    return T, R, C, F

def static_maternal_healthcare(N, b=0.1):
    # WIQL paper: maternal healthcare setting - static case
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
                [0, 0.4, 0.6], # S state action 0
                [0, 0.45, 0.55]  # S state action 1
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
                [0, 0.4, 0.6], # S state action 0
                [0, 0.45, 0.55]  # S state action 1
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
                [0, 0.4, 0.6], # S state action 0
                [0, 0.45, 0.55]  # S state action 1
            ]
        ]
    )

    r = [0, 1, 2]
    T = []
    R = []
    F = []

    for _ in range(int(0.2 * N)):
        T.append(T1)
        R.append(r)
        F.append([1, 0, 0])

    for _ in range(int(0.2 * N)):
        T.append(T2)
        R.append(r)
        F.append([0, 1, 0])

    for _ in range(int(0.6 * N)):
        T.append(T3)
        R.append(r)
        F.append([0, 0, 1])
    
    T = np.array(T)
    R = np.array(R)
    C = np.array([0, 1])
    F = np.array(F)

    return T, R, C, F


def rmab_context_env(N, type_dist):
    T1 = np.array([[[0.99, 0.01],  # for state B action 0
                    [0.5, 0.5]],  # for state B action 1

                   [[0.3, 0.7],  # for state G action 0
                    [0.1, 0.9]]  # for state G action 1
                   ])

    T = np.array([T1 for _ in range(N)])

    # rewards
    R = []
    for i in type_dist:

        r = [0, random.uniform(0.7, 1)]
        R += [r for _ in range(i)]
        # for j in range(i):

    R = np.array(R)

    print(R.shape)

    # R = np.array([[-1,0,0,1] for _ in range(N)]) # rewards
    C = np.array([0, 1])
    B = N/2

    # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

    return T, R, C, B


def rmab_context_diffT_env(N, type_dist):
    T = []
    done = False
    for i in type_dist:
        if done:
            p = random.uniform(0.8, 0.85)

            t = np.array([[[p, 1-p],  # for state B action 0
                           [p, 1-p]],  # for state B action 1

                          [[p, 1-p],  # for state G action 0
                           [p, 1-p]]  # for state G action 1
                          ])
            T += [t for _ in range(i)]
        else:
            done = True
            iv = 0.005
            p = random.uniform(0.8, 0.9)
            pp = p + iv

            t = np.array([[[p, 1-p],  # for state B action 0
                           [1-p, p]],  # for state B action 1

                          [[pp, 1-pp],  # for state G action 0
                           [1-pp, pp]]  # for state G action 1
                          ])
            T += [t for _ in range(i)]

    T = np.array(T)
    R = np.array([np.array([0, 1]) for _ in range(N)])  # rewards
    C = np.array([0, 1])
    B = N/2

    # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

    return T, R, C, B


def rmab_context_features(N, K=1):
    F = np.array(sorted(np.random.random_sample(N)))
    T = []
    R = []

    for i in range(N):
        fi = F[i]
        # ben1 = np.random.uniform(0.01, 0.1)
        # ben2 = np.random.uniform(0.01, 0.1)
        ben1 = 0.1
        ben2 = 0.1

        p000 = fi + 0.5
        if p000 > 0.9:
            p000 = 0.9
        p010 = p000 - ben1

        p100 = fi + 0.2
        if p100 > 0.7:
            p100 = 0.7

        p110 = p100 - ben2

        t = np.array([[[p000, 1-p000],  # for state B action 0
                       [p010, 1-p010]],  # for state B action 1

                      [[p100, 1-p100],  # for state G action 0
                       [p110, 1-p110]]  # for state G action 1
                      ])
        # print(t)
        r = [0, 1]

        T.append(t)
        R.append(r)

    T = np.array(T)
    R = np.array(R)
    C = np.array([0, 1])
    F = F / K

    return T, R, C, F


def aditya(N, type_dist, epsilon=0.005, shift=0.05):
    fname = '../data/patient_T_matrices.npy'
    real = np.load(fname)

    T = []
    # Passive action transition probabilities
    penalty_pass_00 = 0
    penalty_pass_11 = 0

    # Active action transition probabilities
    benefit_act_00 = 0
    benefit_act_11 = 0

    choices = np.random.choice(
        np.arange(real.shape[0]), len(type_dist), replace=True)

    i = 0
    while i < len(type_dist):
        choice = choices[i]
        T_base = np.zeros((2, 2))
        T_base[0, 0] = real[choice][0]
        T_base[1, 1] = real[choice][1]
        T_base[0, 1] = 1 - T_base[0, 0]
        T_base[1, 0] = 1 - T_base[1, 1]

        T_base = smooth_real_probs(T_base, epsilon)

        # Patient responds well to call
        # will subtract from prob of staying 0,0
        benefit_act_00 = np.random.uniform(low=0., high=shift)
        # will add to prob of staying 1,1
        benefit_act_11 = benefit_act_00 + np.random.uniform(low=0., high=shift)
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition

        # Patient does well on their own, low penalty for not calling
        # will sub from prob of staying 1,1
        penalty_pass_11 = np.random.uniform(low=0., high=shift)
        # will add to prob of staying 0,0
        penalty_pass_00 = penalty_pass_11+np.random.uniform(low=0., high=shift)

        T_pass = np.copy(T_base)
        T_act = np.copy(T_base)

        T_act[0, 0] = max(0, T_act[0, 0] - benefit_act_00)
        T_act[1, 1] = min(1, T_act[1, 1] + benefit_act_11)

        T_pass[0, 0] = min(1, T_pass[0, 0] + penalty_pass_00)
        T_pass[1, 1] = max(0, T_pass[1, 1] - penalty_pass_11)

        T_pass[0, 1] = 1 - T_pass[0, 0]
        T_pass[1, 0] = 1 - T_pass[1, 1]

        T_act[0, 1] = 1 - T_act[0, 0]
        T_act[1, 0] = 1 - T_act[1, 1]

        T_pass = epsilon_clip(T_pass, epsilon)
        T_act = epsilon_clip(T_act, epsilon)

        # import pdb
        # pdb.set_trace()

        # t = np.array([T_pass[0, :], T_act[0, :], T_pass[1, :], T_act[1, :]])
        t = np.array([[T_pass[0, :], T_act[0, :]],
                     [T_pass[1, :], T_act[1, :]]])

        if not verify_T_matrix(t):
            print("T matrix invalid\n", t)
            raise ValueError()

        T += [t for _ in range(type_dist[i])]

        i += 1

    T = np.array(T)
    R = np.array([np.array([0, 1]) for _ in range(N)])  # rewards
    C = np.array([0, 1])
    B = N/2

    # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

    return T, R, C, B


def rmab_context_diffT_diffR_env(N, type_dist):

    # T = np.array([T1 for _ in range(N)])
    T = []
    # rewards
    R = []
    for i in type_dist:

        r = [0, random.uniform(0.7, 1)]

        t000 = random.uniform(0.9, 0.99)
        t101 = random.uniform(0.4, 0.6)
        t010 = random.uniform(0.4, 0.6)
        t111 = random.uniform(0.9, 0.999)

        t = np.array([[[t000, 1-t000],  # for state B action 0
                       [1-t101, t101]],  # for state B action 1

                      [[t010, 1-t010],  # for state G action 0
                          [1-t111, t111]]  # for state G action 1
                      ])
        print(t)

        R += [r for _ in range(i)]
        T += [t for _ in range(i)]

        # for j in range(i):

    R = np.array(R)
    T = np.array(T)

    # print(R.shape)

    # R = np.array([[-1,0,0,1] for _ in range(N)]) # rewards
    C = np.array([0, 1])
    B = N/2

    # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

    return T, R, C, B


def arpita_sim1(N):

    T1 = np.array([[[0.5, 0, 0, 0.5],  # for state 0 action 0
                    [0.5, 0.5, 0, 0]],  # for state 0 action 1

                   [[0.5, 0.5, 0, 0],  # for state 1 action 0
                    [0, 0.5, 0.5, 0]],  # for state 1 action 1

                   [[0, 0.5, 0.5, 0],  # for state 2 action 0
                    [0, 0, 0.5, 0.5]],  # for state 2 action 1

                   [[0, 0, 0.5, 0.5],  # for state 3 action 0
                    [0.5, 0, 0, 0.5]]  # for state 3 action 1
                   ])

    T = np.array([T1 for _ in range(N)])

    R = np.array([[-1, 0, 0, 1] for _ in range(N)])  # rewards
    C = np.array([0, 1])
    B = N/5

    print(R.shape)
    # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

    return T, R, C, B


# first simulation environment from arpita's paper extended to multiaction
# by introducing a 3rd action that is half cost, but half as effective
# def arpita_sim1_multiaction(N, extra_states=0):

# 	T1 = np.array(  [[[0.5, 0.0, 0.0, 0.5], #for state 0 action 0
# 				      [0.0, 0.0, 0.0, 1.0],#for state 0 action 1
# 	                  [0.5, 0.5, 0.0, 0.0]],#for state 0 action 2

# 		              [[0.5, 0.5, 0.0, 0.0],#for state 1 action 0
# 		               [0.0, 1.0, 0.0, 0.0],#for state 1 action 1
# 		               [0.0, 0.5, 0.5, 0.0]],#for state 1 action 2

# 		              [[0.0, 0.5, 0.5, 0.0],#for state 2 action 0
# 		               [0.0, 0.0, 1.0, 0.0],#for state 2 action 1
# 		               [0.0, 0.0, 0.5, 0.5]],#for state 2 action 2

# 		              [[0.0, 0.0, 0.5, 0.5],#for state 3 action 0
# 		               [1.0, 0.0, 0.0, 0.0], #for state 3 action 1
# 		               [0.5, 0.0, 0.0, 0.5]] #for state 3 action 2
# 		              ])

# 	states_built_in = T1.shape[0]
# 	total_states = states_built_in + extra_states
# 	total_actions = T1.shape[1]

# 	if extra_states > 0:
# 		T1_old = T1
# 		T1 = np.zeros((total_states,total_actions,total_states))
# 		T1[:states_built_in,:,:states_built_in] = T1_old
# 		T1[states_built_in:,:,0] = 1

# 	T = np.array([T1 for _ in range(N)])
# 	rewards = [-1,0,0,1]
# 	for _ in range(extra_states):
# 		rewards.append(0)

# 	R = np.array([rewards for _ in range(N)]) # rewards
# 	# C = np.array([0, 1, 3])
# 	C = np.array([0, 1, 2])
# 	B = 3*N/5

# 	# prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

# 	return T, R, C, B


# first simulation environment from arpita's paper extended to multiaction
# by introducing a 3rd action that is half cost, but half as effective
def arpita_sim1_multiaction(N, extra_states=0):
    base_middle_states = 2
    total_middle_states = base_middle_states + extra_states

    total_states = 2 + total_middle_states
    NUM_ACTIONS = 3  # This needs to be hard coded right now.

    T1 = np.zeros((total_states, NUM_ACTIONS, total_states))

    # Reconstructing this but with a loop that creates an arbitrary number of middle states
    # T1 = np.array(  [[[0.5, 0.0, 0.0, 0.5], #for state 0 action 0
    # 			        [0.0, 0.0, 0.0, 1.0],#for state 0 action 1
    #                   [0.5, 0.5, 0.0, 0.0]],#for state 0 action 2

    # 	              [[0.5, 0.5, 0.0, 0.0],#for state 1 action 0
    # 	               [0.0, 1.0, 0.0, 0.0],#for state 1 action 1
    # 	               [0.0, 0.5, 0.5, 0.0]],#for state 1 action 2

    # 	              [[0.0, 0.5, 0.5, 0.0],#for state 2 action 0
    # 	               [0.0, 0.0, 1.0, 0.0],#for state 2 action 1
    # 	               [0.0, 0.0, 0.5, 0.5]],#for state 2 action 2

    # 	              [[0.0, 0.0, 0.5, 0.5],#for state 3 action 0
    # 	               [1.0, 0.0, 0.0, 0.0], #for state 3 action 1
    # 	               [0.5, 0.0, 0.0, 0.5]] #for state 3 action 2
    # 	              ])

    # Set the first and last state transitions
    s = 0
    T1[s, 0, 0] = 0.5
    T1[s, 0, -1] = 0.5

    T1[s, 1, -1] = 1.0

    T1[s, 2, 0] = 0.5
    T1[s, 2, 1] = 0.5

    s = -1
    T1[s, 0, -2] = 0.5
    T1[s, 0, -1] = 0.5

    T1[s, 1, 0] = 1.0

    T1[s, 2, 0] = 0.5
    T1[s, 2, -1] = 0.5

    # 	              [[0.5, 0.5, 0.0, 0.0],#for state 1 action 0
    # 	               [0.0, 1.0, 0.0, 0.0],#for state 1 action 1
    # 	               [0.0, 0.5, 0.5, 0.0]],#for state 1 action 2

    # 	              [[0.0, 0.5, 0.5, 0.0],#for state 2 action 0
    # 	               [0.0, 0.0, 1.0, 0.0],#for state 2 action 1
    # 	               [0.0, 0.0, 0.5, 0.5]],#for state 2 action 2

    # now loop to create the middle states
    # offset by one because we already did the 0 state
    for s in range(1, 1+total_middle_states):
        T1[s, 0, s-1] = 0.5
        T1[s, 0, s] = 0.5

        T1[s, 1, s] = 1.0

        T1[s, 2, s] = 0.5
        T1[s, 2, s+1] = 0.5

    T = np.array([T1 for _ in range(N)])
    # [-1,0,0,1]
    rewards = np.zeros(total_states)
    rewards[0] = -1
    rewards[-1] = 1

    R = np.array([rewards for _ in range(N)])  # rewards
    # C = np.array([0, 1, 3])
    C = np.array([0, 1, 2])
    B = 3*N/5

    return T, R, C, B


# Like the original get greedy but has small chance of return to first state
def get_greedy_eng12(S, A, check_function=check_T_strict, recovery_prob=0.05):

    T = np.zeros((S, A, S))

    must_act_to_move = np.zeros(S)
    must_act_to_move[1] = 1

    s = 0
    # First action must be 1 else you get sent to the dead state
    T[s, 1] = must_act_to_move
    for a in [0]+list(range(2, A)):
        must_act_to_move = np.zeros(S)
        must_act_to_move[-1] = 1
        T[s, a] = must_act_to_move

    for s in range(1, S-2):
        for a in range(A):
            if s == a:
                increasing_action_cost = np.zeros(S)
                increasing_action_cost[s+1] = 1
                T[s, a] = increasing_action_cost

            else:
                must_act_to_move = np.zeros(S)
                must_act_to_move[-1] = 1
                T[s, a] = must_act_to_move

    s = S-2
    for a in range(A):
        if s == a:
            increasing_action_cost = np.zeros(S)
            increasing_action_cost[s] = 1
            T[s, a] = increasing_action_cost

        else:
            must_act_to_move = np.zeros(S)
            must_act_to_move[-1] = 1
            T[s, a] = must_act_to_move

    s = -1
    # All actions (including a=0) give small chance of returning to start state
    probs = np.zeros(S)
    probs[0] = recovery_prob
    probs[s] = 1 - recovery_prob
    for a in range(A):
        T[s, a] = probs

    return T

# Like the original get reliable but has small chance of return to first state


def get_reliable_eng12(S, A, check_function=check_T_strict, recovery_prob=0.05):

    T = np.zeros((S, A, S))

    must_act_to_move = np.zeros(S)
    must_act_to_move[1] = 1

    s = 0
    # First action must be 1 or higher else or you get locked in the dead state
    T[s, 1] = must_act_to_move

    must_act_to_move = np.zeros(S)
    must_act_to_move[-1] = 1
    T[s, 0] = must_act_to_move

    for a in range(2, A):
        must_act_to_move = np.zeros(S)
        must_act_to_move[1] = 1
        T[s, a] = must_act_to_move

    # after that, all actions keep you locked in state 1
    for s in range(1, S-1):
        for a in range(1, A):
            if s == 1:
                always_stay = np.zeros(S)
                always_stay[1] = 1
                T[s, a] = always_stay
            else:
                always_stay = np.zeros(S)
                always_stay[-1] = 1
                T[s, a] = always_stay

        # and whenever you don't act, go to dead state
        a = 0
        always_fail = np.zeros(S)
        always_fail[-1] = 1
        T[s, a] = always_fail

    s = -1
    # All actions (including a=0) give small chance of returning to start state
    probs = np.zeros(S)
    probs[0] = recovery_prob
    probs[s] = 1 - recovery_prob
    for a in range(A):
        T[s, a] = probs

    return T

# Like eng7 but with small recovery prob
# increasing cost for increasing reward, but burns out eventually
# vs. turn on at beginning or never get rewar


def get_eng12_experiment(N, A, percent_greedy, REWARD_BOUND, recovery_prob=0.05):

    S = A+1
    T = np.zeros((N, S, A, S))

    num_greedy = int(N*percent_greedy)
    for i in range(num_greedy):
        # cliff, no ladder
        T[i] = get_greedy_eng12(
            S, A, recovery_prob=recovery_prob, check_function=no_check)
        # print("getting nonrecov")
    for i in range(num_greedy, N):
        # cliff with ladder
        T[i] = get_reliable_eng12(
            S, A, recovery_prob=recovery_prob, check_function=no_check)
        # print("getting good on their own")

    R = np.array([np.arange(S) for _ in range(N)])

    # all states after the first have reward of 2
    R[:, 2:-1] = 2

    # Dead state has no reward
    R[:, -1] = 0

    C = np.arange(A)

    return T, R, C


# Like the original get greedy but has small chance of return to first state
def get_greedy_eng13(S, A, recovery_prob=0.00):

    T = np.zeros((S, A, S))

    # First action must be >= 1 to go to state 2, else you get sent to the dead state
    # this arm never uses state 1
    must_act_to_move = np.zeros(S)
    s = 0
    s_next = 2
    must_act_to_move[s_next] = 1
    for a in range(1, A):
        T[s, a] = must_act_to_move

    a = 0
    go_to_dead_state = np.zeros(S)
    go_to_dead_state[-1] = 1
    T[s, a] = go_to_dead_state

    # filler so it's a complete mdp
    s = 1
    go_to_dead_state = np.zeros(S)
    go_to_dead_state[-1] = 1
    for a in range(A):
        T[s, a] = go_to_dead_state

    s = 2
    s_next = 2
    a = 2
    larger_action_cost = np.zeros(S)
    larger_action_cost[s_next] = 1
    T[s, a] = larger_action_cost

    go_to_dead_state = np.zeros(S)
    go_to_dead_state[-1] = 1
    for a in [0, 1]:
        T[s, a] = go_to_dead_state

    s = -1
    # All actions (including a=0) give small chance of returning to start state
    probs = np.zeros(S)
    probs[0] = recovery_prob
    probs[s] = 1 - recovery_prob
    for a in range(A):
        T[s, a] = probs

    return T

# Like the original get reliable but has small chance of return to first state


def get_reliable_eng13(S, A, recovery_prob=0.05):

    T = np.zeros((S, A, S))

    # First action must be >= 1 to go to state 1, else you get sent to the dead state
    # this arm never uses state 2
    must_act_to_move = np.zeros(S)
    s = 0
    s_next = 1
    must_act_to_move[s_next] = 1
    for a in range(1, A):
        T[s, a] = must_act_to_move

    a = 0
    go_to_dead_state = np.zeros(S)
    go_to_dead_state[-1] = 1
    T[s, a] = go_to_dead_state

    s = 1
    s_next = 1
    a = 1
    stay_in_state_1 = np.zeros(S)
    stay_in_state_1[s_next] = 1
    T[s, a] = stay_in_state_1

    # make it so expensive action also keeps you in state 1
    a = 2
    T[s, a] = stay_in_state_1

    # no action sends to dead states
    go_to_dead_state = np.zeros(S)
    go_to_dead_state[-1] = 1
    for a in [0]:
        T[s, a] = go_to_dead_state

    # filler so it's a complete mdp
    s = 2
    go_to_dead_state = np.zeros(S)
    go_to_dead_state[-1] = 1
    for a in range(A):
        T[s, a] = go_to_dead_state

    s = -1
    # All actions (including a=0) give small chance of returning to start state
    probs = np.zeros(S)
    probs[0] = recovery_prob
    probs[s] = 1 - recovery_prob
    for a in range(A):
        T[s, a] = probs

    return T

# Like eng7 but with small recovery prob
# increasing cost for increasing reward, but burns out eventually
# vs. turn on at beginning or never get rewar


def get_eng13_experiment(N, A, percent_greedy, REWARD_BOUND, recovery_prob=0.05):

    S = 4
    A = 3
    T = np.zeros((N, S, A, S))

    num_greedy = int(N*percent_greedy)
    for i in range(num_greedy):
        # cliff, no ladder
        T[i] = get_greedy_eng13(S, A, recovery_prob=recovery_prob)
        # print("getting nonrecov")
    for i in range(num_greedy, N):
        # cliff with ladder
        T[i] = get_reliable_eng13(S, A, recovery_prob=recovery_prob)
        # print("getting good on their own")

    R = np.array([np.zeros(S) for _ in range(N)])

    # set rewards
    R[:, 0] = 0
    R[:, 1] = 1
    R[:, 2] = 1.1

    # Dead state has no reward
    R[:, -1] = 0

    C = np.arange(A)

    return T, R, C


# Patient needs to be acted on all the time to stay adhering
def get_needy_eng14_bu(S, A, recovery_prob=0.05, fail_prob_passive=0.95, fail_prob_act=0.05):

    T = np.zeros((S, A, S))

    # patient needs to be acted on all the time to stay adhering
    S = 2

    s = 0
    for a in range(0, A):
        pr = recovery_prob*(a+1)/(a+2)
        T[s, a] = [1.0-pr, pr]

    s = 1
    for a in range(0, A):
        # pf = 1 - ((1-fail_prob_act)+(1-fail_prob_act)*(a-0.5)/(a+0.5))
        if a == 0:
            pf = fail_prob_passive

        T[s, a] = [pf, 1-pf]

    return T

# These patients improve when acted on, drop when not


def get_reliable_eng14_bu(S, A, p00passive=0.75, p01active=0.75, p10passive=0.25, p11active=0.85, base_action_effect=0.25):

    T = np.zeros((S, A, S))

    # patient does better when acted on, worse when not
    S = 2

    s = 0
    a = 0
    T[s, a] = [p00passive, 1-p00passive]

    for a in range(1, A):
        # pr = p01active + base_action_effect*(a+1)/(a+2)
        pr = p01active + base_action_effect*(a-1)/(a)
        T[s, a] = [1.0-pr, pr]

    s = 1
    a = 0
    T[s, a] = [p10passive, 1-p10passive]

    for a in range(1, A):
        pr = min(1, p11active + p11active*(a-0.5)/(a+0.5))
        T[s, a] = [1.0-pr, pr]

    return T


# Patient needs to be acted on all the time to stay adhering
def get_needy_eng14(S, A, recovery_prob=0.05, fail_prob_passive=0.95, fail_prob_act=0.05):

    T = np.zeros((S, A, S))

    # patient needs to be acted on all the time to stay adhering
    S = 2

    s = 0
    for a in range(0, A):
        pr = recovery_prob*(a+1)/(a+2)
        T[s, a] = [1.0-pr, pr]

    s = 1
    for a in range(0, A):
        pf = 0
        if a == 1:
            pf = 1-(0.75/2+0.25)
        elif a == 2:
            pf = 1-0.99
        if a == 0:
            pf = fail_prob_passive

        T[s, a] = [pf, 1-pf]

    return T

# These patients improve when acted on, drop when not


def get_reliable_eng14(S, A, p00passive=0.75, p01active=0.75, p10passive=0.25, p11active=0.85, base_action_effect=0.25):

    T = np.zeros((S, A, S))

    # patient does better when acted on, worse when not
    S = 2

    s = 0
    a = 0
    T[s, a] = [p00passive, 1-p00passive]

    for a in range(1, A):
        pr = 0
        if a == 1:
            pr = 0.75 + 0.125
        elif a == 2:
            pr = 0.99
        T[s, a] = [1.0-pr, pr]

    s = 1
    a = 0
    T[s, a] = [p10passive, 1-p10passive]

    for a in range(1, A):
        pr = 0
        if a == 1:
            pr = 0.75 + 0.125
        elif a == 2:
            pr = 0.99
        T[s, a] = [1.0-pr, pr]

    return T

# Like eng7 but with small recovery prob
# increasing cost for increasing reward, but burns out eventually
# vs. turn on at beginning or never get rewar
# def get_eng14_experiment(N, A, percent_greedy, REWARD_BOUND,
# 	recovery_prob=0.4, fail_prob_passive=0.75, fail_prob_act=0.05,
# 	p00passive = 0.95, p01active=0.75, p10passive=0.25, p11active=0.5, base_action_effect=0.1):


def get_eng14_experiment(N, A, percent_greedy, REWARD_BOUND,
                         recovery_prob=0.4, fail_prob_passive=0.75, fail_prob_act=0.4,
                         p00passive=0.95, p01active=0.60, p10passive=0.25, p11active=0.6, base_action_effect=0.4):

    S = 2
    A = 3
    T = np.zeros((N, S, A, S))

    num_greedy = int(N*percent_greedy)
    for i in range(num_greedy):
        # cliff, no ladder
        T[i] = get_needy_eng14(S, A, recovery_prob=recovery_prob,
                               fail_prob_passive=fail_prob_passive, fail_prob_act=fail_prob_act)
        # print("getting nonrecov")
    for i in range(num_greedy, N):

        # cliff with ladder
        T[i] = get_reliable_eng14(S, A, p00passive=p00passive, p01active=p01active,
                                  p10passive=p10passive, p11active=p11active, base_action_effect=base_action_effect)
        # print("getting good on their own")

    R = np.array([np.zeros(S) for _ in range(N)])

    # set rewards
    R[:, 0] = 0
    R[:, 1] = 1

    print(T)
    # 1/0
    C = np.arange(A)
    np.save('eng_14_N%s_rewards.npy' % N, R)
    # 1/0

    return T, R, C


# Patient needs to be acted on all the time to stay adhering
def get_needy_eng15_history(history_length, A, base_states, recovery_prob=0.05, fail_prob_passive=0.95, fail_prob_act=0.05):

    HL = history_length
    num_states = 2**HL

    state_effect = 0.05
    weights = np.linspace(1, 2, history_length)
    state_weights = weights*state_effect/weights.sum()

    T = np.zeros((num_states, A, num_states))

    # now to enumerate all possible states transitions...
    historied_states = [seq for seq in itertools.product((0, 1), repeat=HL)]
    state_dict = dict([(state, i) for i, state in enumerate(historied_states)])

    for current_state in historied_states:

        current_state_ind = state_dict[current_state]

        active_base_state = current_state[-1]

        current_state_shifted_by_1 = list(current_state[1:])
        for next_base_state in base_states:
            next_historied_state = tuple(
                current_state_shifted_by_1+[next_base_state])
            next_state_ind = state_dict[next_historied_state]
            state_bonus = state_weights.dot(current_state)
            for a in range(0, A):
                if active_base_state == 0 and next_base_state == 0:
                    pr = recovery_prob*(a+1)/(a+2)
                    pr += state_bonus
                    T[current_state_ind, a, next_state_ind] = 1 - pr
                elif active_base_state == 0 and next_base_state == 1:
                    pr = recovery_prob*(a+1)/(a+2)
                    pr += state_bonus
                    T[current_state_ind, a, next_state_ind] = pr

                elif active_base_state == 1 and next_base_state == 0:

                    pf = 0
                    if a == 1:
                        pf = 1-(0.75/2+0.25)
                    elif a == 2:
                        pf = 1-0.94
                    if a == 0:
                        pf = fail_prob_passive
                    pf -= state_bonus

                    T[current_state_ind, a, next_state_ind] = pf

                elif active_base_state == 1 and next_base_state == 1:

                    pf = 0
                    if a == 1:
                        pf = 1-(0.75/2+0.25)
                    elif a == 2:
                        pf = 1-0.94
                    if a == 0:
                        pf = fail_prob_passive
                    pf -= state_bonus

                    T[current_state_ind, a, next_state_ind] = 1-pf

    return T


# https://www-jstor-org.ezp-prod1.hul.harvard.edu/stable/23033358?sid=primo&seq=1#metadata_info_tab_contents
def lambdafunc(s, a):

    phi = np.random.rand()*4.25 + 0.75
    alpha = np.random.rand()*0.45 + 1.05
    xi = (11**alpha - (s+1)**alpha)*(s+1)**(-alpha+1)

    return a/(a + phi)*xi

# https://www-jstor-org.ezp-prod1.hul.harvard.edu/stable/23033358?sid=primo&seq=1#metadata_info_tab_contents


def mufunc(s, a):

    phi = np.random.rand()*4.25 + 0.75
    eta = s

    return phi/(a + phi)*eta


def plate_reward_pwlc():
    r = []
    for i in range(11):
        if i <= 4:
            r.append(0)
        elif i >= 5 and i <= 8:
            r.append((i-4)/5)
        elif i == 9 or i == 10:
            r.append(1)
    return r


def plate_reward_concave(A):
    r = []
    for i in range(A):
        r.append(i/(i+1))
    return r


def get_plate(S, A):

    T = np.zeros((S, A, S))

    T[0, 0, 0] = 1

    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            if s < S-1:
                T[s, a, s+1] = lambdafunc(s, a)
            if s > 0:
                T[s, a, s-1] = mufunc(s, a)

            T[s, a] = T[s, a]/T[s, a].sum()

    return T

# glazebrook spinning plate experiment
# https://www-jstor-org.ezp-prod1.hul.harvard.edu/stable/23033358?sid=primo&seq=1#metadata_info_tab_contents


def get_spinningplate_experiment(N, REWARD_BOUND):

    A = 4
    S = 11
    T = np.zeros((N, S, A, S))
    for i in range(N):
        # cliff, no ladder
        T[i] = get_plate(S, A)
        # print("getting good on their own")

    R = np.array([plate_reward_pwlc() for _ in range(N)])

    # R = np.cumsum(R,axis=1)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    print(R)
    print(T)
    # 1/0

    # # all states after the first have reward of 2
    # R[:,2:-1] = 2

    # # Dead state has no reward
    # R[:,-1] = 0

    C = np.arange(A)
    B = A

    return T, R, C, B


def collapse_history_matrix(T, history_length, shorter_length, longer_states, shorter_states):
    num_s_shorter = 2**shorter_length
    T_shorter = np.zeros((num_s_shorter, num_s_shorter))

    shorter_ind_list = []
    num_shorter_states = len(shorter_states)
    num_longer_states = len(longer_states)

    for j in range(num_shorter_states):
        shorter_state = shorter_states[j]
        inds_to_combine = []

        for i in range(num_longer_states):
            longer_state = longer_states[i]
            suffix = longer_state[-shorter_length:]
            # print(suffix)
            if np.array_equal(suffix, shorter_state):
                inds_to_combine.append(i)
                # print(long_state)
        # print()
        shorter_ind_list.append(inds_to_combine)

    shorter_ind_list = np.array(shorter_ind_list)

    for i in range(num_shorter_states):
        inds_i = shorter_ind_list[i]
        for j in range(num_shorter_states):
            inds_j = shorter_ind_list[j]
            collapsed_row = T[inds_i].sum(axis=0)
            collapsed_entry = collapsed_row[inds_j].sum()
            T_shorter[i, j] = collapsed_entry
    # print(T_shorter)
    for i in range(num_shorter_states):
        row = T_shorter[i]
        T_shorter[i] = row/row.sum()
    # print(T_shorter)

    return T_shorter


#   [0,0][0,1][1,0][1,1]
# [
#  [[0.2, 0.2, 0.3, 0.3]]
#  [[0.3, 0.3, 0.2, 0.2]]
#  [[0.1, 0.4, 0.5, 0.0]]
#  [[0.4, 0.1, 0.1, 0.4]]
# ]
#
#


def advance_simulation(current_state, T):
    next_state = np.argmax(np.random.multinomial(1, T[current_state]))
    return next_state


def simulate_start_states(T, history_length):

    longer_states = np.array(
        [seq for seq in itertools.product((0, 1), repeat=history_length)])

    current_states = np.ones(T.shape[0], dtype=object)

    for shorter_length in range(1, history_length):
        shorter_states = [seq for seq in itertools.product(
            (0, 1), repeat=shorter_length)]
        shorter_state_dict = dict([(state, i)
                                  for i, state in enumerate(shorter_states)])

        for arm in range(T.shape[0]):
            T_arm = collapse_history_matrix(
                T[arm, :, 0], history_length, shorter_length, longer_states, shorter_states)
            current_state_string = current_states[arm]
            if type(current_state_string) == int:
                current_state_string = (current_state_string,)
            current_state_index = shorter_state_dict[current_state_string]
            next_state_index = advance_simulation(current_state_index, T_arm)
            next_state_string = shorter_states[next_state_index]
            new_state_tuple = tuple(
                list(current_state_string) + list(next_state_string[-1:]))
            current_states[arm] = new_state_tuple

    longer_state_dict = dict([(tuple(state), i)
                             for i, state in enumerate(longer_states)])
    current_states = np.array([longer_state_dict[current_state]
                              for current_state in current_states])

    return current_states


def get_tb_patients_with_history(N, num_actions, history_length, action_diff_mult, REWARD_BOUND):

    fname = '../data/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy' % (
        N, num_actions, history_length, action_diff_mult)
    T = np.load(fname)

    # Get one reward if most recent state is adhering, 0 otherwise
    R_single = np.ones(2**history_length)
    inds = np.arange(2**(history_length-1))
    inds *= 2
    R_single[inds] -= 1

    R = np.array([R_single for _ in range(N)])

    # R = np.cumsum(R,axis=1)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    # print(R)
    # print(T)
    # 1/0

    # # all states after the first have reward of 2
    # R[:,2:-1] = 2

    # # Dead state has no reward
    # R[:,-1] = 0

    start_states = simulate_start_states(T, history_length)

    C = np.arange(num_actions)
    B = N/5

    return T, R, C, B, start_states


def get_tb_patients_plus_needy_with_history(N, num_actions, history_length, action_diff_mult, REWARD_BOUND,
                                            percent_greedy, file_root=None,
                                            recovery_prob=0.4, fail_prob_passive=0.75, fail_prob_act=0.4):

    base_states = [0, 1]
    HL = history_length
    S = 2**HL

    # fname = '../data/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy'%(N, num_actions, history_length, action_diff_mult)
    fname = file_root+'/data/frequentist/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy' % (
        N, num_actions, history_length, action_diff_mult)
    T = np.load(fname)

    num_greedy = int(N*percent_greedy)
    for i in range(num_greedy):
        # cliff, no ladder
        T[i] = get_needy_eng15_history(history_length, num_actions, base_states, recovery_prob=recovery_prob,
                                       fail_prob_passive=fail_prob_passive, fail_prob_act=fail_prob_act)

    # Get one reward if most recent state is adhering, 0 otherwise
    R_single = np.ones(2**history_length)
    inds = np.arange(2**(history_length-1))
    inds *= 2
    R_single[inds] -= 1

    R = np.array([R_single for _ in range(N)])

    # R = np.cumsum(R,axis=1)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    # print(R)
    # print(T)
    # 1/0

    # # all states after the first have reward of 2
    # R[:,2:-1] = 2

    # # Dead state has no reward
    # R[:,-1] = 0

    start_states = simulate_start_states(T, history_length)

    C = np.arange(num_actions)
    B = N/4
    print(T)
    np.save('eng_15_S%s_N%s_rewards.npy' % (S, N), R)

    # 1/0
    return T, R, C, B, start_states
