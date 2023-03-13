from email import policy
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
import rmab_ql
import simulation_environments
import matplotlib.pyplot as plt
import sys
import random

TS_policies = [21, 22, 25, 101, 102]

def update_counts(actions, state_log, counts):
    for arm, a in enumerate(actions):
        a = int(a)
        s = state_log[arm, 0]
        sprime = state_log[arm, 1]
        counts[arm, s, a, sprime] += 1
    return counts


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


def takeAction(current_states, T, actions, random_stream):

    N = len(current_states)

    # Get next state
    # print(current_states)
    # print(T.shape)
    next_states = np.zeros(current_states.shape)
    
    for i in range(N):

        current_state = int(current_states[i])

        next_state = np.argmax(random_stream.multinomial(
            1, T[i, current_state, int(actions[i]), :]))
        next_states[i] = next_state

        # if current_state != next_state:
        #     print(i, current_state, int(actions[i]), next_state, T[i, current_state, int(actions[i]), :])

    # print(next_states)

    return next_states


def getActions(N, T_hat, R, C, B, k, features=None, seed=None, valid_action_combinations=None, combined_state_dict=None, current_state=None,
               optimal_policy=None, policy_option=0, gamma=0.95, indexes=None, type_dist=None,
               output_data=None, True_T=None, qlearning_objects=None, learning_random_stream=None, t=None,
               action_to_play=None, current_state_log=None):

    if policy_option == 0:
        # Nobody
        return np.zeros(N)

    elif policy_option == 1:
        # Everybody
        return np.ones(N)

    elif policy_option == 2:
        # Random
        # Randomly pick from all valid options
        choices = np.arange(valid_action_combinations.shape[0])
        choice = np.random.choice(choices)
        return valid_action_combinations[choice]

    # Fast random, inverse weighted
    elif policy_option == 3:

        actions = np.zeros(N, dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(N), N, replace=False)
        for arm in process_order:

            # select an action at random
            num_valid_actions_left = len(C[C <= B-current_action_cost])
            # p = 1/(C[C<=B-current_action_cost]+1)
            # p = p/p.sum()
            p = None
            a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
            current_action_cost += C[a]
            # if the next selection takes us over budget, break
            if current_action_cost > B:
                break

            actions[arm] = a
        
        return actions

    elif policy_option == 4:
        # optimal MDP

        state_tup = tuple(current_state.astype(int))
        print(state_tup)

        state_ind = combined_state_dict[state_tup]
        action_index = optimal_policy[state_ind]
        return valid_action_combinations[action_index]

    elif policy_option == 10:

        actions = np.zeros(N, dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(N), N, replace=False)
        for arm in process_order:

            # select an action at random
            num_valid_actions_left = len(C[C <= B-current_action_cost])
            # p = 1/(C[C<=B-current_action_cost]+1)
            # p = p/p.sum()
            p = None
            a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
            current_action_cost += C[a]
            # if the next selection takes us over budget, break
            if current_action_cost > B:
                break

            actions[arm] = a

        return actions

    elif policy_option == 21:
        indexes_per_state = []
        for i in range(len(current_state)):
            whittle = get_whittle_indices(T_hat[i], gamma, T.shape[1])
            indexes_per_state.append([0, whittle[current_state[i]]])
        indexes_per_state = np.array(indexes_per_state)
        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)
        actions = np.argmax(decision_matrix, axis=1)
        if not (decision_matrix.sum(axis=1) <= 1).all():
            raise ValueError("More than one action per person" +
                             str(actions)+str(decision_matrix))

        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B:
            raise ValueError("Over budget")
        return actions

    # Q-learning
    elif policy_option in [31, 32, 33, 34, 35, 36, 61, 
                            62, 63, 64, 65, 66, 41, 42, 
                            43, 44, 91, 92, 93, 94, 101, 102]:
        ql_object = qlearning_objects['ql_object']
        actions = np.zeros(N, dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon
        if ql_object.check_random(t, random_stream=learning_random_stream):
            # print('Doing a random')
            if N <= 10:
                return getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                  policy_option=3, combined_state_dict=combined_state_dict,
                                  indexes=indexes, output_data=output_data, True_T=True_T,
                                  t=t, qlearning_objects=qlearning_objects)
            else:
                return getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                  policy_option=3, combined_state_dict=combined_state_dict,
                                  indexes=indexes, output_data=output_data, True_T=True_T,
                                  qlearning_objects=qlearning_objects)

        # print('exploiting')

        # # otherwise act greedily
        indexes_per_state = ql_object.get_indexes()

        # indexes_per_state = indexes_per_state.cumsum(axis=1)

        # print('indexes')
        # print(indexes_per_state)
        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)
        # print("knapsack time:",time.time() - start)

        # print(decision_matrix)
        actions = np.argmax(decision_matrix, axis=1)
        if not (decision_matrix.sum(axis=1) <= 1).all():
            raise ValueError("More than one action per person" +
                             str(actions)+str(decision_matrix))

        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B:
            raise ValueError("Over budget")

        return actions

    elif policy_option in [54]:

        sys.exit(0)

    elif policy_option in [71, 72, 73, 74]:
        ql_object = qlearning_objects['ql_object_context']

        #if ql_object.check_random(t, random_stream=learning_random_stream):
        if t < 0:
            # if False:
            if N <= 10:
                actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                     policy_option=3, combined_state_dict=combined_state_dict,
                                     indexes=indexes, output_data=output_data, True_T=True_T,
                                     t=t, qlearning_objects=qlearning_objects)

            else:
                actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                     policy_option=3, combined_state_dict=combined_state_dict,
                                     indexes=indexes, output_data=output_data, True_T=True_T,
                                     qlearning_objects=qlearning_objects)

            # print(actions)
            return actions

        indexes_per_state = ql_object.get_whittleindexes().copy()
        actions = np.zeros(N)
        chosen = np.zeros(N)
        explore = np.zeros(N)
        for i in range(len(explore)):
            #print(i)
            if(ql_object.check_random(t, random_stream=learning_random_stream)):
                explore[i] = 1
            if(np.sum(explore) >=  B):
                break
        print(B)
        print(np.sum(explore))
        #print('\n')
        #print(current_state_log)

        sorted_indexes = np.dstack(np.unravel_index(np.argsort(
            indexes_per_state.ravel()), indexes_per_state.shape))[0]

        idx = [0]
        c = 0
        for i in type_dist:
            c += i
            idx.append(c)

        #if(t <= 17 or t == 25 or t == 27 or t == 35):
        #    print('t = ' + str(t))
        #    print(ql_object.get_whittleindexes())
        #    print("arm, state, action, count")
        for tup in reversed(sorted_indexes):
            arm = tup[0]
            state = tup[1]
            action = tup[2]
            temp = np.array([*range(idx[arm], idx[arm + 1])])
            #print("start and end indices", idx[arm], idx[arm + 1])
            np.random.shuffle(temp)
            count = 0
            for i in temp:
                if current_state_log[i] == state:
                    count = count + 1
            #print(arm, state, action, count)

        #print(sorted_indexes)
        #exit()

        #print(C)
        frac = np.sum(explore)*1.0/N
        B_exploit = (int)((1-frac)*B)
        B_explore = (int) (frac*B)
        #print("before:")
        #print(B_exploit)
        #print(B_explore)
        if(B_exploit+B_explore < B):
            B_exploit = B - B_explore
        #print("after:")
        #print(B_exploit)
        #print(B_explore)
        budget = 0
        done = False
        for tup in reversed(sorted_indexes):
            arm = tup[0]
            state = tup[1]
            action = tup[2]
            
            #print("selecting from users in cluster = %d, state = %d, action = %d", arm, state, action)
            temp = np.array([*range(idx[arm], idx[arm + 1])])
            #print("start and end indices", idx[arm], idx[arm + 1])
            np.random.shuffle(temp)

            for i in temp:
                if(explore[i] == 1):
                    continue
                if current_state_log[i] == state:
                    if(chosen[i] == 1):
                        continue
                    budget += C[action]
                    #if(t==45):
                    #    print("C[action]", C[action])
                    #    print("action", action)
                    #    print(budget)
                    if budget > B_exploit:
                        done = True
                        break

                    actions[i] = action
                    if(C[action] > 0):
                        chosen[i] = 1
                    #print("action for user i is", i, action)
        explore_indices = np.where(explore == 1)[0]
        print("explore_length")
        print(len(explore_indices))
        selected_to_explore = random.sample(list(explore_indices), B_explore)
        #print(len(selected_to_explore))
        #print("actions before explore")
        #print(np.sum(actions))
        for i in range(len(selected_to_explore)):
            #if(actions[i]==1):
                #print("already selected")
            actions[i] = 1
            
        #print(np.sum(actions))
            #if done:
                #print("tup")
                #print(tup)
                #print("action")
                #print(action)
                #break
        
        """if(np.sum(actions) < B):
            print(np.sum(actions))
            print("whittle indices")
            print(ql_object.get_whittleindexes())
            print("sorted clusters by whittle indices")
            print(sorted_indexes)
        """
        
        #if(t==45):
        #    print(budget)
        #    print(np.sum(actions))
            
            
        return actions

    elif policy_option == 80:
        actions = np.zeros(N, dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(N), N, replace=False)
        for arm in process_order:

            # select an action at random

            num_valid_actions_left = len(
                C[arm][1:][C[arm][1:] <= B-current_action_cost])

            p = None

            a = 0
            try:
                a = np.random.choice(
                    np.arange(num_valid_actions_left), 1, p=p)[0]
                current_action_cost += C[arm][1:][a]
            except:
                actions[arm] = -(1 + a)
                break
            # if the next selection takes us over budget, break
            if current_action_cost > B:
                actions[arm] = -(1 + a)
                break

            actions[arm] = 1 + a

        return actions

    elif policy_option in [81, 82, 83, 84]:

        ql_object = qlearning_objects['ql_object_context']

        # with prob epsilon, explore randomly
        # This call will also decay epsilon
        if ql_object.check_random(t, random_stream=learning_random_stream):
            # print('Doing a random')
            if N <= 10:
                actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                     policy_option=80, combined_state_dict=combined_state_dict,
                                     indexes=indexes, output_data=output_data, True_T=True_T,
                                     t=t, qlearning_objects=qlearning_objects)

            else:
                actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                     policy_option=80, combined_state_dict=combined_state_dict,
                                     indexes=indexes, output_data=output_data, True_T=True_T,
                                     qlearning_objects=qlearning_objects)

            # print(actions)
            return actions

        # print('exploiting')

        # # otherwise act greedily
        indexes_per_state = ql_object.get_indexes().copy()
        # indexes_per_state = indexes_per_state.cumsum(axis=1)

        # sys.exit(0)
        # print("budget: ", B)
        # print('indexes')
        # print(indexes_per_state)

        # print(C.transpose())
        # print(B)

        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)
        # print("knapsack time:",time.time() - start)

        # print(decision_matrix)
        actions = np.argmax(decision_matrix, axis=1)
        if not (decision_matrix.sum(axis=1) <= 1).all():
            raise ValueError("More than one action per person" +
                             str(actions)+str(decision_matrix))

        payment = 0
        for i in range(N):
            payment += C[i][actions[i]]
            if actions[i] != 0:
                indexes_per_state[i, :] = -10000

        # print(actions)

        if not payment <= B:
            raise ValueError("Over budget")
        else:
            residual_payment = B - payment
            while residual_payment > 0:
                # print(indexes_per_state)
                arm, act = np.unravel_index(
                    np.argmax(indexes_per_state, axis=None), indexes_per_state.shape)
                # print(arm, act)
                if C[arm][act] / type_dist[arm] > residual_payment:
                    indexes_per_state[arm, :] = -10000
                    continue
                if actions[arm] == 0:
                    actions[arm] = -act
                break

        # print(actions)
        return actions


def thompson_sampling(N, T_shape, priors, counts, random_stream):

    T_hat = np.zeros(T_shape)
    for i in range(N):
        for j in range(T_hat.shape[1]):
            for k in range(T_hat.shape[2]):
                params = priors[i, j, k, :] + counts[i, j, k, :]
                T_hat[i, j, k, 0] = random_stream.beta(params[0], params[1])
                T_hat[i, j, k, 1] = 1 - T_hat[i, j, k, 0]
    return T_hat

def simulateAdherence(N, L, T, R, C, B, k, policy_option, optimal_policy=None, combined_state_dict=None,
                      action_logs={}, features=None, cumulative_state_log=None,
                      seedbase=None, savestring='trial', learning_mode=False,
                      world_random_seed=None, learning_random_seed=None, verbose=False,
                      file_root=None, gamma=0.95, type_dist=None,
                      output_data=None, start_state=None, do_plot=None,
                      pname=None, LC=None, kappa=0, beta=0.4, lud=2,
                      qinit=0, eqdist=0, nknown=0):

    learning_random_stream = np.random.RandomState()
    if learning_mode > 0:
        learning_random_stream.seed(learning_random_seed)

    world_random_stream = np.random.RandomState()
    world_random_stream.seed(world_random_seed)

    qvalues = None

    # set up thompson sampling
    T_hat = None
    priors = np.ones(T.shape)
    counts = np.zeros(T.shape)

    qlearning_objects = {}

    # Extensions of Arpita AAMAS QL
    eps = 0.75
    gamma = gamma
    alpha = 0.4
    n_states = T.shape[1]
    n_actions = T.shape[2]
    ql_type = 3
    if policy_option > 40 and policy_option % 10 == 1:
        ql_type = 0
    elif policy_option > 40 and policy_option % 10 == 2:
        ql_type = 1
    elif policy_option > 40 and policy_option % 10 == 3:
        ql_type = 2
    elif policy_option > 40 and policy_option % 10 == 4:
        ql_type = 3
    # print("ql_type", ql_type)

    if qinit != 0:
        qinit = np.max(R)/(1-gamma)
    if eqdist != 0:
        eqdist = True
    else:
        eqdist = False

    ql_object = rmab_ql.RMABQL(N, k, eps, alpha, gamma,
                               L, n_states, n_actions, initial_exploration=False,
                               eps_decay=True, ql_type=ql_type,
                               qinit=qinit, eqdist=eqdist, nknown=nknown)
    qlearning_objects['ql_object'] = ql_object

    print("type_dist", type_dist)
    if type_dist is not None and len(type_dist) != 0:
        if policy_option >= 81 and policy_option <= 84:
            N_super = len(type_dist)
            # 0-25, 25-50, 50-75, 75-100 (%) of people adhering
            super_states = 4
            super_actions = 2                     # 0, 1

            ql_object_context = rmab_ql.RMABQL(N_super, k, eps, alpha, gamma,
                                               L, super_states, super_actions, initial_exploration=False,
                                               eps_decay=True, ql_type=ql_type,
                                               qinit=qinit, eqdist=eqdist, nknown=nknown)
            qlearning_objects['ql_object_context'] = ql_object_context

        elif policy_option >= 71 and policy_option <= 74:
            print(type_dist)
            N_super = len(type_dist)
            print(N_super)
            ql_object_context = rmab_ql.RMABQL(N_super, k, eps, alpha, gamma,
                                               L, n_states, n_actions, initial_exploration=False,
                                               eps_decay=True, ql_type=ql_type,
                                               qinit=qinit, eqdist=eqdist, nknown=nknown)
            qlearning_objects['ql_object_context'] = ql_object_context

    state_log = np.zeros((N, L), dtype=int)
    actions_record = np.zeros((N, L-1))

    if policy_option >= 80 and policy_option < 90:
        super_state_log = np.zeros((N_super, L), dtype=int)
        super_actions_record = np.zeros((N_super, L-1))

    if action_logs is not None:
        action_logs[policy_option] = []

    indexes = np.zeros((N, C.shape[0]))

    print('Running simulation w/ policy: %s' % policy_option)

    # TODO: Change to make this random start state
    # state_log[:,0]=T.shape[1]-1
    if start_state is not None:
        state_log[:, 0] = start_state
    else:
        state_log[:, 0] = 1

    if policy_option >= 80 and policy_option < 90:
        super_indexes = np.zeros((N_super, C.shape[0]))
        determine_super_states(state_log, super_state_log,
                               0, type_dist, super_states)

    #######  Run simulation #######
    print('Running simulation w/ policy: %s' % policy_option)
    # make array of nan to initialize observations
    learning_modes = ['no_learning', 'Thompson sampling']
    # print("Learning mode:", learning_modes[learning_mode])
    print("Policy:", pname[policy_option])

    # if problem size is small enough, enumerate all valid actions
    # to use for random exploration
    # else, we will use "fast" random which has some issues right now
    valid_action_combinations = None
    if policy_option in [2, 41, 42, 43, 44] and N <= 5:
        options = np.array(list(product(np.arange(C.shape[0]), repeat=N)))
        valid_action_combinations = utils.list_valid_action_combinations(
            N, C, B, options)

    qvalues_log = []
    file = open("policy.txt", "a")
    file2 = open("frequency.txt", "a")
    if policy_option >= 71:
        file.write('--------------------------------- \n')
        file.write('This is the policy for a new seed \n')
        file.write('--------------------------------- \n')
        file2.write('--------------------------------- \n')
        file2.write('This is the policy for a new seed \n')
        file2.write('--------------------------------- \n')

    
    for t in tqdm.tqdm(range(1, L)):
        #print("Round: %s"%t)

        #print("t = ")
        #print(t)
        st = time.time()
        T_hat = None
        if learning_mode == 0:
            T_hat = T
        elif learning_mode == 1 and policy_option in TS_policies:
            # Thompson sampling
            T_hat = thompson_sampling(N, T.shape, priors, counts, random_stream=learning_random_stream)
        
        if t == 20:
            T = []
            #5 blocks of 20
            """T1 = np.array(
                [
                    [
                        [0.5203357, 0.378954164, 0.100710136], # G state action 0
                        [0.66576087, 0.298913043, 0.035326087]  # G state action 1
                    ],
                    [
                        [0.385703064, 0.422623723, 0.191673213], # M state action 0
                        [0.625, 0.311111111, 0.063888889]  # M state action 1
                    ],
                    [
                        [0.22160149, 0.43575419, 0.34264432], # B state action 0
                        [0.495575221, 0.398230088, 0.10619469]  # B state action 1
                    ],
                ]
            )

            T2 = np.array(
                [
                    [
                        [0.755652519, 0.217770726, 0.026576755], # G state action 0
                        [0.849462366, 0.144393241, 0.006144393]  # G state action 1
                    ],
                    [
                        [0.638924456, 0.298335467, 0.062740077], # M state action 0
                        [0.779569892, 0.204301075, 0.016129032]  # M state action 1
                    ],
                    [
                        [0.481481481, 0.361111111, 0.157407407], # B state action 0
                        [0.515151515, 0.454545455, 0.03030303]  # B state action 1
                    ],
                ]
            )

            T3 = np.array(
                [
                    [
                        [0.648440121, 0.290841999, 0.06071788], # G state action 0
                        [0.798387097, 0.186827957, 0.014784946]  # G state action 1
                    ],
                    [
                        [0.507014028, 0.372745491, 0.120240481], # M state action 0
                        [0.730245232, 0.226158038, 0.04359673]  # M state action 1
                    ],
                    [
                        [0.357995227, 0.398568019, 0.243436754], # B state action 0
                        [0.641304348, 0.315217391, 0.043478261]  # B state action 1
                    ],
                ]
            )

            T4 = np.array(
                [
                    [
                        [0.672443674, 0.284228769, 0.043327556], # G state action 0
                        [0.769123783, 0.219749652, 0.011126565]  # G state action 1
                    ],
                    [
                        [0.593724859, 0.333065165, 0.073209976], # M state action 0
                        [0.739413681, 0.2247557, 0.035830619]  # M state action 1
                    ],
                    [
                        [0.45631068, 0.36407767, 0.17961165], # B state action 0
                        [0.6, 0.383333333, 0.016666667]  # B state action 1
                    ],
                ]
            )"""
            #3blocks of 20
            T1 = np.array(
                [
                    [
                        [0.478761699, 0.39524838, 0.125989921], # G state action 0
                        [0.672043011, 0.290322581, 0.037634409]  # G state action 1
                    ],
                    [
                        [0.381458967, 0.40881459, 0.209726444], # M state action 0
                        [0.506666667, 0.376666667, 0.116666667]  # M state action 1
                    ],
                    [
                        [0.216216216, 0.378378378, 0.405405405], # B state action 0
                        [0.439490446, 0.382165605, 0.178343949]  # B state action 1
                    ],
                ]
            )

            T2 = np.array(
                [
                    [
                        [0.75545492, 0.216138329, 0.028406752], # G state action 0
                        [0.7925117, 0.185647426, 0.021840874]  # G state action 1
                    ],
                    [
                        [0.625298329, 0.303102625, 0.071599045], # M state action 0
                        [0.697916667, 0.244791667, 0.057291667]  # M state action 1
                    ],
                    [
                        [0.376712329, 0.465753425, 0.157534247], # B state action 0
                        [0.647058824, 0.235294118, 0.117647059]  # B state action 1
                    ],
                ]
            )

            T3 = np.array(
                [
                    [
                        [0.6256852, 0.304228661, 0.070086139], # G state action 0
                        [0.726973684, 0.251644737, 0.021381579]  # G state action 1
                    ],
                    [
                        [0.410778443, 0.404191617, 0.18502994], # M state action 0
                        [0.570093458, 0.343457944, 0.086448598]  # M state action 1
                    ],
                    [
                        [0.215384615, 0.395384615, 0.389230769], # B state action 0
                        [0.394736842, 0.468421053, 0.136842105]  # B state action 1
                    ],
                ]
            )

            T4 = np.array(
                [
                    [
                        [0.713816869, 0.25041024, 0.035772891], # G state action 0
                        [0.789256198, 0.190082645, 0.020661157]  # G state action 1
                    ],
                    [
                        [0.612208259, 0.324057451, 0.063734291], # M state action 0
                        [0.72875817, 0.251633987, 0.019607843]  # M state action 1
                    ],
                    [
                        [0.569060773, 0.337016575, 0.093922652], # B state action 0
                        [0.630434783, 0.326086957, 0.043478261]  # B state action 1
                    ],
                ]
            )

            for _ in range(int(0.21 * N)):
                T.append(T1)

            for _ in range(int(0.214 * N)):
                T.append(T2)

            for _ in range(int(0.305 * N)):
                T.append(T3)

            for _ in range(int(0.271 * N)):
                T.append(T4)
            T = np.array(T)

            T_hat = T
            
        elif t == 40:
            T = []
            #100 block
            """T1 = np.array(
                [
                    [
                        [0.613319566, 0.31285493, 0.073825503], # G state action 0
                        [0.758835759, 0.216216216, 0.024948025]  # G state action 1
                    ],
                    [
                        [0.489741302, 0.369313113, 0.140945584], # M state action 0
                        [0.66798419, 0.288537549, 0.043478261]  # M state action 1
                    ],
                    [
                        [0.352941176, 0.411764706, 0.235294118], # B state action 0
                        [0.635294118, 0.329411765, 0.035294118]  # B state action 1
                    ],
                ]
            )

            T2 = np.array(
                [
                    [
                        [0.778454511, 0.203654359, 0.017891131], # G state action 0
                        [0.833580981, 0.151560178, 0.014858841]  # G state action 1
                    ],
                    [
                        [0.718130312, 0.236543909, 0.045325779], # M state action 0
                        [0.778443114, 0.203592814, 0.017964072]  # M state action 1
                    ],
                    [
                        [0.554216867, 0.301204819, 0.144578313], # B state action 0
                        [0.75, 0.25, 0]  # B state action 1
                    ],
                ]
            )

            T3 = np.array(
                [
                    [
                        [0.731421884, 0.227911059, 0.040667057], # G state action 0
                        [0.794047619, 0.195238095, 0.010714286]  # G state action 1
                    ],
                    [
                        [0.582524272, 0.330097087, 0.087378641], # M state action 0
                        [0.775641026, 0.208333333, 0.016025641]  # M state action 1
                    ],
                    [
                        [0.454935622, 0.446351931, 0.098712446], # B state action 0
                        [0.672131148, 0.278688525, 0.049180328]  # B state action 1
                    ],
                ]
            )

            T4 = np.array(
                [
                    [
                        [0.716715734, 0.248609748, 0.034674518], # G state action 0
                        [0.810844893, 0.182849937, 0.00630517]  # G state action 1
                    ],
                    [
                        [0.62601626, 0.325203252, 0.048780488], # M state action 0
                        [0.807971014, 0.177536232, 0.014492754]  # M state action 1
                    ],
                    [
                        [0.578947368, 0.348684211, 0.072368421], # B state action 0
                        [0.742857143, 0.228571429, 0.028571429]  # B state action 1
                    ],
                ]
            )"""
            #3blocks of 20
            T1 = np.array(
                [
                    [
                        [0.580725907, 0.335419274, 0.083854819], # G state action 0
                        [0.687943262, 0.260047281, 0.052009456]  # G state action 1
                    ],
                    [
                        [0.411962932, 0.425442291, 0.162594777], # M state action 0
                        [0.577854671, 0.349480969, 0.07266436]  # M state action 1
                    ],
                    [
                        [0.286419753, 0.439506173, 0.274074074], # B state action 0
                        [0.454545455, 0.386363636, 0.159090909]  # B state action 1
                    ],
                ]
            )

            T2 = np.array(
                [
                    [
                        [0.780254777, 0.196257962, 0.023487261], # G state action 0
                        [0.820469799, 0.17114094, 0.008389262]  # G state action 1
                    ],
                    [
                        [0.708841463, 0.254573171, 0.036585366], # M state action 0
                        [0.720430108, 0.252688172, 0.02688172]  # M state action 1
                    ],
                    [
                        [0.5, 0.3, 0.2], # B state action 0
                        [0.769230769, 0.153846154, 0.076923077]  # B state action 1
                    ],
                ]
            )

            T3 = np.array(
                [
                    [
                        [0.661411411, 0.277777778, 0.060810811], # G state action 0
                        [0.735988201, 0.237463127, 0.026548673]  # G state action 1
                    ],
                    [
                        [0.466135458, 0.394422311, 0.139442231], # M state action 0
                        [0.579634465, 0.347258486, 0.07310705]  # M state action 1
                    ],
                    [
                        [0.286353468, 0.447427293, 0.266219239], # B state action 0
                        [0.572649573, 0.307692308, 0.11965812]  # B state action 1
                    ],
                ]
            )

            T4 = np.array(
                [
                    [
                        [0.742666667, 0.235333333, 0.022], # G state action 0
                        [0.814661134, 0.16735823, 0.017980636]  # G state action 1
                    ],
                    [
                        [0.642927794, 0.302670623, 0.054401583], # M state action 0
                        [0.711538462, 0.261538462, 0.026923077]  # M state action 1
                    ],
                    [
                        [0.483870968, 0.403225806, 0.112903226], # B state action 0
                        [0.741935484, 0.258064516, 0]  # B state action 1
                    ],
                ]
            )

            for _ in range(int(0.21 * N)):
                T.append(T1)

            for _ in range(int(0.214 * N)):
                T.append(T2)

            for _ in range(int(0.305 * N)):
                T.append(T3)

            for _ in range(int(0.271 * N)):
                T.append(T4)
            T = np.array(T)

            T_hat = T
        
        """
        #100 block
        elif t == 60:
            T = []
            T1 = np.array(
                [
                    [
                        [0.661214953, 0.288785047, 0.05], # G state action 0
                        [0.783733826, 0.205175601, 0.011090573]  # G state action 1
                    ],
                    [
                        [0.552093476, 0.357351509, 0.090555015], # M state action 0
                        [0.738396624, 0.244725738, 0.016877637]  # M state action 1
                    ],
                    [
                        [0.446078431, 0.387254902, 0.166666667], # B state action 0
                        [0.529411765, 0.411764706, 0.058823529]  # B state action 1
                    ],
                ]
            )

            T2 = np.array(
                [
                    [
                        [0.811123149, 0.17659805, 0.012278801], # G state action 0
                        [0.868035191, 0.123167155, 0.008797654]  # G state action 1
                    ],
                    [
                        [0.751219512, 0.208130081, 0.040650407], # M state action 0
                        [0.825174825, 0.167832168, 0.006993007]  # M state action 1
                    ],
                    [
                        [0.563636364, 0.363636364, 0.072727273], # B state action 0
                        [0.6875, 0.25, 0.0625]  # B state action 1
                    ],
                ]
            )

            T3 = np.array(
                [
                    [
                        [0.774897681, 0.200818554, 0.024283765], # G state action 0
                        [0.840390879, 0.153094463, 0.006514658]  # G state action 1
                    ],
                    [
                        [0.656787763, 0.265774379, 0.077437859], # M state action 0
                        [0.783464567, 0.204724409, 0.011811024]  # M state action 1
                    ],
                    [
                        [0.448275862, 0.41954023, 0.132183908], # B state action 0
                        [0.55, 0.375, 0.075]  # B state action 1
                    ],
                ]
            )

            T4 = np.array(
                [
                    [
                        [0.752959502, 0.224922118, 0.02211838], # G state action 0
                        [0.823529412, 0.168067227, 0.008403361]  # G state action 1
                    ],
                    [
                        [0.671701913, 0.282980866, 0.045317221], # M state action 0
                        [0.837944664, 0.138339921, 0.023715415]  # M state action 1
                    ],
                    [
                        [0.539215686, 0.411764706, 0.049019608], # B state action 0
                        [0.655172414, 0.344827586, 0]  # B state action 1
                    ],
                ]
            )
            for _ in range(int(0.21 * N)):
                T.append(T1)

            for _ in range(int(0.214 * N)):
                T.append(T2)

            for _ in range(int(0.305 * N)):
                T.append(T3)

            for _ in range(int(0.271 * N)):
                T.append(T4)
            T = np.array(T)

            T_hat = T


        else:
            T = []
            T1 = np.array(
                [
                    [
                        [0.705641492, 0.262056415, 0.032302093], # G state action 0
                        [0.829824561, 0.159649123, 0.010526316]  # G state action 1
                    ],
                    [
                        [0.609489051, 0.323600973, 0.066909976], # M state action 0
                        [0.717948718, 0.256410256, 0.025641026]  # M state action 1
                    ],
                    [
                        [0.463235294, 0.382352941, 0.154411765], # B state action 0
                        [0.8, 0.166666667, 0.033333333]  # B state action 1
                    ],
                ]
            )

            T2 = np.array(
                [
                    [
                        [0.82731671, 0.15630815, 0.01637514], # G state action 0
                        [0.882440476, 0.114583333, 0.00297619]  # G state action 1
                    ],
                    [
                        [0.760589319, 0.222836096, 0.016574586], # M state action 0
                        [0.878504673, 0.112149533, 0.009345794]  # M state action 1
                    ],
                    [
                        [0.543478261, 0.391304348, 0.065217391], # B state action 0
                        [0.727272727, 0.272727273, 0]  # B state action 1
                    ],
                ]
            )

            T3 = np.array(
                [
                    [
                        [0.789242591, 0.192371021, 0.018386389], # G state action 0
                        [0.862767154, 0.132733408, 0.004499438]  # G state action 1
                    ],
                    [
                        [0.71192053, 0.247240618, 0.040838852], # M state action 0
                        [0.76754386, 0.206140351, 0.026315789]  # M state action 1
                    ],
                    [
                        [0.615384615, 0.288461538, 0.096153846], # B state action 0
                        [0.708333333, 0.25, 0.041666667]  # B state action 1
                    ],
                ]
            )

            T4 = np.array(
                [
                    [
                        [0.7658831, 0.215057179, 0.01905972], # G state action 0
                        [0.840101523, 0.154822335, 0.005076142]  # G state action 1
                    ],
                    [
                        [0.694063927, 0.276255708, 0.029680365], # M state action 0
                        [0.800884956, 0.181415929, 0.017699115]  # M state action 1
                    ],
                    [
                        [0.611111111, 0.277777778, 0.111111111], # B state action 0
                        [0.666666667, 0.285714286, 0.047619048]  # B state action 1
                    ],
                ]
            )
            for _ in range(int(0.21 * N)):
                T.append(T1)

            for _ in range(int(0.214 * N)):
                T.append(T2)

            for _ in range(int(0.305 * N)):
                T.append(T3)

            for _ in range(int(0.271 * N)):
                T.append(T4)
            T = np.array(T)

            T_hat = T"""
        

        if policy_option == 21:
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:, t-1],
                                 optimal_policy=optimal_policy, type_dist=type_dist,
                                 policy_option=policy_option, combined_state_dict=combined_state_dict, gamma=gamma,
                                 indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                                 t=t, qlearning_objects=qlearning_objects, current_state_log=state_log[:, t-1].reshape(-1))

        # Epsilon greedy part
        if policy_option >= 71 and policy_option <= 74:
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:, t-1],
                                 optimal_policy=optimal_policy, type_dist=type_dist,
                                 policy_option=policy_option, combined_state_dict=combined_state_dict, gamma=gamma,
                                 indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                                 t=t, qlearning_objects=qlearning_objects, current_state_log=state_log[:, t-1].reshape(-1))

            # sas = process_superarm_state_action(state_log[:,t-1].reshape(-1), actions, type_dist)

        elif policy_option >= 80 and policy_option <= 84:

            C_super = []
            for i in type_dist:
                C_super.append([j*i for j in C])

            C_super = np.array(C_super)

            actions_super = getActions(N_super, T_hat, R, C_super, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:, t-1],
                                       optimal_policy=optimal_policy, type_dist=type_dist,
                                       policy_option=policy_option, combined_state_dict=combined_state_dict, gamma=gamma,
                                       indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                                       t=t, qlearning_objects=qlearning_objects)
            actions = np.zeros(N, dtype=int)

            # print(actions_super)

            convert_super_actions(C, B, type_dist, N_super,
                                  actions_super, actions)
            super_actions_record[:, t-1] = actions_super

            # print(actions_super)
            # print(actions)

        elif policy_option >= 90 and policy_option <= 94:
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:, t-1],
                                 optimal_policy=optimal_policy, type_dist=type_dist,
                                 policy_option=policy_option-50, combined_state_dict=combined_state_dict, gamma=gamma,
                                 indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                                 t=t, qlearning_objects=qlearning_objects)

        elif policy_option in [54]:
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations,
                                 features=features, seed=world_random_seed,
                                 current_state=state_log[:, t -
                                                         1], optimal_policy=optimal_policy, type_dist=type_dist,
                                 policy_option=policy_option, combined_state_dict=combined_state_dict, gamma=gamma,
                                 indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                                 t=t, qlearning_objects=qlearning_objects)

        else:
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:, t-1],
                                 optimal_policy=optimal_policy, type_dist=type_dist,
                                 policy_option=policy_option, combined_state_dict=combined_state_dict, gamma=gamma,
                                 indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                                 t=t, qlearning_objects=qlearning_objects)

        actions_record[:, t-1] = actions

        if action_logs is not None:
            action_logs[policy_option].append(actions.astype(int))

        # TODO: Modify T_hat to estimated value of T_hat (sliding window, etc.)
        state_log[:, t] = takeAction(
            state_log[:, t-1].reshape(-1), T, actions, random_stream=world_random_stream)


        # if policy_option >= 71 and policy_option <= 74:
        #     R_super, visits = determine_super_reward_sas(state_log[:, t-1].reshape(-1),
        #                                                  state_log[:, t].reshape(-1), actions, type_dist, R, n_states, n_actions)

        if policy_option >= 81 and policy_option <= 84:
            determine_super_states(
                state_log, super_state_log, t, type_dist, super_states)
            R_super = determine_super_reward(
                state_log, super_state_log, t, R, type_dist, super_states)

        # update q_learning
        if policy_option in [41, 42, 43, 44]:
            ql_object.qlearn(actions, state_log[:, t-1:], R, t, C)
            qvalues = ql_object.getQvalues()
            qvalues_log.append(qvalues.copy())


        if policy_option in [71, 72, 73, 74]:
            C_super = np.array([0, 1])
            # ql_object_context.qlearn_super(R_super, visits, C_super)
            ql_object_context.qlearn_super(actions, state_log[:, t-1:], R, type_dist)
            qvalues = ql_object_context.getQvalues()
            qvalues_log.append(qvalues.copy())
            to_print=np.nonzero(actions)
            #print('selected players:')
            file.write('t = ')
            file.write(str(t))
            file.write('\n')
            counters_cluster = np.zeros(4)
            for a in to_print:
                file.write(str(a)+', ')
            #    print(a)
            #    print(' ')
            #print('\n')
            for a in range(len(actions)):
                if(a < 210 and actions[a] == 1):
                    counters_cluster[0] = counters_cluster[0] + 1
                elif(a < 424 and actions[a] == 1):
                    counters_cluster[1] = counters_cluster[1] + 1
                elif(a < 729 and actions[a] == 1):
                    counters_cluster[2] = counters_cluster[2] + 1
                elif(actions[a] == 1):
                    counters_cluster[3] = counters_cluster[3] + 1

            for a in range(len(counters_cluster)):
                file2.write(str(counters_cluster[a])+'\t')
            file2.write('\n')
            file.write('\n')
            
        if policy_option in [81, 82, 83, 84]:
            ql_object_context.qlearn(
                actions_super, super_state_log[:, t-1:], R_super, t, C_super)
            qvalues = ql_object_context.getQvalues()
            
            qvalues_log.append(qvalues.copy())

        if policy_option in [91, 92, 93, 94]:
            if t <= lud or nknown != 0:
                ql_object.qlearn_lipschitz(
                    actions, state_log[:, t-1:], R, t, C, features, LC, kappa, beta)
            else:
                ql_object.qlearn(actions, state_log[:, t-1:], R, t, C)
            qvalues = ql_object.getQvalues()
            qvalues_log.append(qvalues.copy())

        
        if policy_option in [31, 32, 33, 34, 35, 36]:
            if policy_option == 36:
                ql_object.qlearn_lipschitz_optimistic(
                    actions, state_log[:, t-1:], R, t, C, features, LC, kappa, beta, M=0.5)
            else:
                ql_object.qlearn_lipschitz_optimistic(
                    actions, state_log[:, t-1:], R, t, C, features, LC, kappa, beta, M=(policy_option%10))
            qvalues = ql_object.getQvalues()
            qvalues_log.append(qvalues.copy())
        
        if policy_option in [61, 62, 63, 64, 65, 66]:
            if policy_option == 66:
                ql_object.qlearn_lipschitz_optimistic_add(
                    actions, state_log[:, t-1:], R, t, C, features, LC, kappa, beta, M=0.5)
            else:
                ql_object.qlearn_lipschitz_optimistic_add(
                    actions, state_log[:, t-1:], R, t, C, features, LC, kappa, beta, M=(policy_option%10))
            qvalues = ql_object.getQvalues()
            qvalues_log.append(qvalues.copy())
        
        if policy_option in [101]:
            # if t <= lud:
            ql_object.qlearn_dabel_lipschitz(
                    actions, state_log[:, t-1:], R, t, C, features, LC, T_hat)
            # else:
            #     ql_object.qlearn(actions, state_log[:, t-1:], R, t, C)
            qvalues = ql_object.getQvalues()
            qvalues_log.append(qvalues.copy())
        
        if policy_option in [102]:
            if t <= lud:
                ql_object.qlearn_dabel_lipschitz(
                    actions, state_log[:, t-1:], R, t, C, features, LC, T_hat)
            else:
                ql_object.qlearn(actions, state_log[:, t-1:], R, t, C)
            qvalues = ql_object.getQvalues()
            qvalues_log.append(qvalues.copy())


        if learning_mode == 1:
            update_counts(actions, state_log[:, t-1:], counts)
            # if t > L-2:
            #     print(T, T_hat)
        
    
    qvalues_log = np.array(qvalues_log)

    if cumulative_state_log is not None:
        cumulative_state_log[policy_option] = np.cumsum(state_log.sum(axis=0))

    # print("Final Indexes")

    return state_log, action_logs, qvalues_log


def determine_super_reward_sas(state_log, next_state_log, actions, type_dist, R, n_states, n_actions):
    R_super = np.zeros((len(type_dist), n_states, n_actions, n_states))
    count = np.zeros((len(type_dist), n_states, n_actions, n_states))

    # s_arm = 0
    # for i in range(len(state_log)):
    #     s = int(state_log[i])
    #     a = int(actions[i])
    #     s_prime = int(next_state_log[i])

        

    #     if i == type_dist[s_arm]:
    #         s_arm += 1

    #     R_super[s_arm, s, a, s_prime] += R[i, s]
    #     count[s_arm, s, a, s_prime] += 1


    # for i in range(len(type_dist)):
    #     for j in range(n_states):
    #         for k in range(n_actions):
    #             for l in range(n_states):
    #                 if count[i, j, k, l] != 0:
    #                     R_super[i, j, k, l] /= count[i, j, k, l]

    return R_super, count


def determine_super_reward(state_log, super_state_log, t, R, type_dist, n_states):
    R_super = np.zeros((len(type_dist), n_states))

    arm = 0
    for i in range(len(type_dist)):
        r = 0
        for j in range(type_dist[i]):
            r += R[arm, state_log[arm, t]]
            arm += 1
        R_super[i, super_state_log[i, t]] = r / type_dist[i]

    return R_super


def determine_super_states(state_log, super_state_log, t, type_dist, n_states):
    blocks = 1./n_states
    p = 0
    for i in range(len(type_dist)):
        s = ((np.sum(state_log[p:p+type_dist[i], t]))/type_dist[i])//blocks
        if s == n_states:
            s -= 1
        super_state_log[i, t] = s
        p += type_dist[i]


def convert_super_actions(C, B, type_dist, N_super, actions_super, actions):
    arm = 0
    cost = 0
    for arm_s in range(N_super):
        if actions_super[arm_s] >= 0:
            for i in range(type_dist[arm_s]):
                actions[arm] = actions_super[arm_s]
                cost += C[actions[arm]]
                arm += 1
        else:
            arm += type_dist[arm_s]

    arm = 0
    for arm_s in range(N_super):
        if actions_super[arm_s] < 0:
            budget_left = B - cost
            randomized_subarms = np.random.choice(
                np.arange(type_dist[arm_s]), type_dist[arm_s], replace=False)

            for sub_arm in randomized_subarms:
                budget_left -= C[-actions_super[arm_s]]
                if budget_left < 0:
                    break

                actions[arm + sub_arm] = -actions_super[arm_s]
                cost += C[actions[arm + sub_arm]]

            arm += type_dist[arm_s]
            actions_super[arm_s] = -actions_super[arm_s]
            break
        else:
            arm += type_dist[arm_s]


def solve_steady(T, n_actions):
    ss_prob = []
    for i in range(len(T)):
        temp = []
        for a in range(n_actions):
            matrix = T[i, :, a]
            a = matrix[0, 0]
            b = matrix[1, 1]
            s0 = (1-b) / (2-(a+b))
            s1 = (1-a) / (2-(a+b))
            ss = [s0, s1]
            temp.append(ss)
        ss_prob.append(temp)
    ss_prob = np.array(ss_prob)
    return ss_prob


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run adherence simulations with various methods.')
    parser.add_argument('-n', '--num_patients', default=2,
                        type=int, help='Number of Processes')
    parser.add_argument('-b', '--budget_frac', default=0.5,
                        type=float, help='Budget per day as fraction of n')
    parser.add_argument('-l', '--simulation_length', default=180,
                        type=int, help='Number of days to run simulation')
    parser.add_argument('-N', '--num_trials', default=5,
                        type=int, help='Number of trials to run')
    parser.add_argument('-S', '--num_states', default=2,
                        type=int, help='Number of states per process')
    parser.add_argument('-A', '--num_actions', default=2,
                        type=int, help='Number of actions per process')
    parser.add_argument('-g', '--discount_factor', default=0.95,
                        type=float, help='Discount factor for MDP solvers')
    parser.add_argument('-nt', '--num_types', default=None,
                        type=int, help="Number of distinct feature arm types")
    parser.add_argument('-td', '--type_dist', nargs='+', default=None,
                        type=int, help="Distribution of types of arms")

    parser.add_argument('-d', '--data', default='real', choices=['rmab_context_features_lipschitz', 'rmab_context_diffT', 'rmab_context_diffT_diffR',
                                                                'rmab_context_features', 'full_random_online', 'arpita_sim1',
                                                                'arpita_sim1_multiaction', 'rmab_context', 'arpita_circulant_dynamics',
                                                                'arpita_healthcare_static', 'arpita_circulant_dynamics_prewards', 'simple_spam_ham', 'simple_spam_ham_2'],
                        type=str, help='Method for generating transition probabilities')

    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-ws', '--world_seed_base', default=None,
                        type=int, help='Base for the random seed')
    parser.add_argument('-ls', '--learning_seed_base',
                        default=None, type=int, help='Base for the random seed')

    parser.add_argument('-f', '--file_root', default='./..', type=str,
                        help='Root dir for experiment (should be the dir containing this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int,
                        help='policy to run, default is all policies')
    parser.add_argument('-tr', '--trial_number',
                        default=None, type=int, help='Trial number')
    parser.add_argument('-sv', '--save_string', default='', type=str,
                        help='special string to include in saved file name')

    parser.add_argument('-lr', '--learning_option', default=0, choices=[0, 1], 
                        type=int, help='0: No Learning (Ground truth known)\n1: Thompson Sampling')
    parser.add_argument('-sts', '--start-state', default=-1,
                        type=int, help='-1: Random, other int: start state for all')
    parser.add_argument('-beta', '--beta', default=0.4, type=float,
                        help='Learning rate beta value')
    parser.add_argument('-lud', '--lud', default=0, type=int,
                        help='Duration to use lipschitz update')
    parser.add_argument('-qinit', '--qinit', default=0, choices=[0,1], type=int,
                        help="Initialize pessimistic or optimistic initialization")
    parser.add_argument('-eqdist', '--eqdist', default=0, choices=[0,1], type=int,
                        help='Update learning rate if distance is same (same arms)')
    parser.add_argument('-nknown', '--nknown', default=0, type=int,
                        help='Number of updates before you start considering (s,a) for lipschitz transfer')
    args = parser.parse_args()

    # File root
    if args.file_root == '.':
        args.file_root = os.getcwd()

    # Save special name
    if args.save_string == '':
        args.save_string = str(
            time.ctime().replace(' ', '_').replace(':', '_'))

    # Policies to run
    if args.policy < 0:
        # policies = [3, 31, 36, 61, 66, 44, 74, 94]
        policies = [0, 3, 74]
    else:
        policies = [args.policy]

    # policy names dict
    pname = {
        0: 'nobody',    2: 'Random',
        3: 'FastRandom',
        4: 'MDP-Optimal',

        10: 'MDPSteady',

        21: 'Whittle-TS',

        31: 'Lipschitz-OptimisticQ-M1',
        32: 'Lipschitz-OptimisticQ-M2',
        33: 'Lipschitz-OptimisticQ-M3',
        36: 'Lipschitz-OptimisticQ-M0.5',

        41: 'WIQL1',
        42: 'WIQL2',
        43: 'WIQL3',
        44: 'WIQL4',

        54: 'UCB-Super',

        61: 'Lipschitz-OptimisticQ-Add-M1',
        62: 'Lipschitz-OptimisticQ-Add-M2',
        63: 'Lipschitz-OptimisticQ-Add-M3',
        66: 'Lipschitz-OptimisticQ-Add-M0.5',

        71: 'WIQL1-Super-SameState',
        72: 'WIQL2-Super-SameState',
        73: 'WIQL3-Super-SameState',
        74: 'WIQL4-Super-SameState',

        81: 'WIQL1-super',
        82: 'WIQL2-super',
        83: 'WIQL3-super',
        84: 'WIQL4-super',

        91: 'Lipschitz-WIQL1',
        92: 'Lipschitz-WIQL2',
        93: 'Lipschitz-WIQL3',
        94: 'Lipschitz-WIQL4',

        100: 'arpita_sim1_optimal',

        101: 'DAbel-Lipschitz',
        102: 'DAbel-Lipschitz-LUD'
    }

    add_to_seed_for_specific_trial = 0

    first_seedbase = np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase = np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial

    first_learning_seedbase = np.random.randint(0, high=100000)
    if args.learning_seed_base is not None:
        first_learning_seedbase = args.learning_seed_base+add_to_seed_for_specific_trial

    N = args.num_patients
    L = args.simulation_length
    k = 0
    savestring = args.save_string
    N_TRIALS = args.num_trials
    LEARNING_MODE = args.learning_option
    beta = args.beta
    lud = args.lud
    F = None
    LC = None

    num_types = args.num_types
    type_dist = args.type_dist

    if args.type_dist is not None and args.num_types is not None:
        assert(len(type_dist) == num_types)

    if args.type_dist is None and args.num_types is not None:
        type_dist = [1 for _ in range(num_types)]

    record_policy_actions = list(pname.keys())

    # for rapid prototyping
    # use this to avoid updating all the function calls when you need to pass in new
    # algo-specific things or return new data
    output_data = {}

    # list because one for each trial
    output_data['hawkins_lambda'] = []
    output_data['lp_index_method_values'] = []


    state_log = dict([(key, []) for key in pname.keys()])
    action_logs = {}
    cumulative_state_log = {}

    mean_reward_log = dict([(key, []) for key in pname.keys()])
    
    qvalues_log = dict([(key, []) for key in pname.keys()])

    window_size = L//2
    mean_reward_log_moving_avg = dict([(key, []) for key in pname.keys()])

    start = time.time()
    file_root = args.file_root

    runtimes = np.zeros((N_TRIALS, len(policies)))

    opt_act = np.zeros((L, 100, 3))
    tot_act = np.zeros((L, 100, 3))
    qv = np.frompyfunc(list, 0, 1)(np.empty((L,100, args.num_states, args.num_actions, 3), dtype=object))
    dqv = np.frompyfunc(list, 0, 1)(np.empty((L,100, args.num_states, 3), dtype=object))

    for i in range(N_TRIALS):

        # do_plot = i==0
        do_plot = False

        # use np global seed for rolling random data, then for random algorithmic choices
        seedbase = first_seedbase + i
        np.random.seed(seed=seedbase)

        # Use world seed only for evolving the world (If two algs
        # make the same choices, should create the same world for same seed)
        world_seed_base = first_world_seedbase + i

        # Use learning seed only for processes involving learning (i.e., exploration vs. exploitation)
        learning_seed_base = first_learning_seedbase + i

        print("Seed is", seedbase)

        T = None
        R = None
        C = None
        B = None
        start_state = None

        # --------------------------------
        if args.data == 'simple_spam_ham':
            T, R, C, F = simulation_environments.simple_spam_ham(N)
            B = args.budget_frac * N
            LC = np.zeros((T.shape[1], T.shape[2]))
            num_types = 4
            type_dist = [int(0.21 * N), int(0.214*N), int(0.305*N), int(0.271*N)]
            args.num_states = T.shape[1]

        if args.data == 'simple_spam_ham_2':
            T, R, C, F = simulation_environments.simple_spam_ham_2(N)
            B = args.budget_frac * N
            LC = np.zeros((T.shape[1], T.shape[2]))
            num_types = 3
            type_dist = [333, 333, 334]
            args.num_states = T.shape[1]

        if args.data == 'arpita_circulant_dynamics':
            T, R, C, F = simulation_environments.circulant_dynamics(N)
            B = args.budget_frac * N
            LC = np.zeros((T.shape[1], T.shape[2]))
            num_types = 3
            type_dist = [N]
            args.num_states = T.shape[1]


        if args.data == 'arpita_circulant_dynamics_prewards':
            T, R, C, F = simulation_environments.circulant_dynamics_prewards(N)
            B = args.budget_frac * N
            LC = np.zeros((T.shape[1], T.shape[2]))
            num_types = 3
            type_dist = [N]
            args.num_states = T.shape[1]

        
        if args.data == 'arpita_healthcare_static':
            T, R, C, F = simulation_environments.static_maternal_healthcare(N)
            B = args.budget_frac * N
            g = args.discount_factor
            LC = np.zeros((T.shape[1], T.shape[2]))
            args.num_states = T.shape[1]
            A = np.array([
                [
                    [1, -g * T[0, 0, 0, 0], 0, -g * T[0, 0, 0, 1], 0, 0],
                    [0, 1-g * T[0, 0, 1, 0], 0, -g * T[0, 0, 1, 1], 0, 0],
                    [0, -g * T[0, 1, 0, 0], 1, 0, 0, -g * T[0, 1, 0, 1]],
                    [0, 0, 0, 1-g * T[0, 1, 1, 1], 0, -g * T[0, 1, 1, 2]],
                    [0, 0, 0, -g * T[0, 2, 0, 1], 1, -g * T[0, 2, 0, 2]],
                    [0, 0, 0, -g * T[0, 2, 1, 1], 0, 1-g * T[0, 2, 1, 2]],
                ],
                [
                    [1, -g * T[(int(0.25 * N)), 0, 0, 0], 0, -g * T[(int(0.25 * N)), 0, 0, 1], 0, 0],
                    [0, 1-g * T[(int(0.25 * N)), 0, 1, 0], 0, -g * T[(int(0.25 * N)), 0, 1, 1], 0, 0],
                    [0, -g * T[(int(0.25 * N)), 1, 0, 0], 1, 0, 0, -g * T[(int(0.25 * N)), 1, 0, 1]],
                    [0, 0, 0, 1-g * T[(int(0.25 * N)), 1, 1, 1], 0, -g * T[(int(0.25 * N)), 1, 1, 2]],
                    [0, 0, 0, -g * T[(int(0.25 * N)), 2, 0, 1], 1, -g * T[(int(0.25 * N)), 2, 0, 2]],
                    [0, 0, 0, -g * T[(int(0.25 * N)), 2, 1, 1], 0, 1-g * T[(int(0.25 * N)), 2, 1, 2]],
                ],
                [
                    [1, -g * T[(int(0.5 * N)), 0, 0, 0], 0, -g * T[(int(0.5 * N)), 0, 0, 1], 0, 0],
                    [0, 1-g * T[(int(0.5 * N)), 0, 1, 0], 0, -g * T[(int(0.5 * N)), 0, 1, 1], 0, 0],
                    [0, -g * T[(int(0.5 * N)), 1, 0, 0], 1, 0, 0, -g * T[(int(0.5 * N)), 1, 0, 1]],
                    [0, 0, 0, 1-g * T[(int(0.5 * N)), 1, 1, 1], 0, -g * T[(int(0.5 * N)), 1, 1, 2]],
                    [0, 0, 0, -g * T[(int(0.5 * N)), 2, 0, 1], 1, -g * T[(int(0.5 * N)), 2, 0, 2]],
                    [0, 0, 0, -g * T[(int(0.5 * N)), 2, 1, 1], 0, 1-g * T[(int(0.5 * N)), 2, 1, 2]],
                ]
            ])
            F1 = [F[0], F[int(0.25 * N)], F[int(0.5 * N)]]
            Y = [0, 0, 1, 1, 2, 2]
            Q = []
            for a in A:
                Q.append(np.linalg.solve(a, Y))
            print(Q)
            for it3 in range(len(Q[0])):
                for it1 in range(len(Q)):
                    for it2 in range(it1 + 1, len(Q)):
                        # print(it1, it2, it3)
                        temp = abs(Q[it1][it3] - Q[it2][it3])/abs(np.linalg.norm(F1[it1] - F1[it2]))
                        LC[it3 // 2, it3 % 2] = max(LC[it3 // 2, it3 % 2], temp)
                        
            num_types = 3
            type_dist = [int(0.2*N), int(0.2*N), int(0.6*N)]
            


        if args.data == 'full_random_online':
            REWARD_BOUND = 1
            start_state = np.zeros(N)
            # T = (N,S,A,S)
            T, R, C, B = simulation_environments.get_full_random_experiment(
                N, args.num_states, args.num_actions, REWARD_BOUND)

        if args.data == 'rmab_context':
            T, R, C, B = simulation_environments.rmab_context_env(N, type_dist)
            # start_state = np.ones(N)
            B = args.budget_frac * N


        if args.data == 'rmab_context_diffT':
            T, R, C, B = simulation_environments.rmab_context_diffT_env(
                N, type_dist)
            # start_state = np.ones(N)
            B = args.budget_frac * N
            # fname = '../logs/T/T_%s_N%s_b%s_L%s_data%s_seed%s_S%s_A%s.dat' % (
            #     savestring, N, args.budget_frac, L, args.data, seedbase, args.num_states, args.num_actions)
            # T[0].tofile(fname)


        if args.data == 'rmab_context_diffT_diffR':
            T, R, C, B = simulation_environments.rmab_context_diffT_diffR_env(
                N, type_dist)
            # start_state = np.ones(N)
            B = args.budget_frac * N
            fname = '../logs/T/T_%s_N%s_b%s_L%s_data%s_seed%s_S%s_A%s.dat' % (
                savestring, N, args.budget_frac, L, args.data, seedbase, args.num_states, args.num_actions)
            T.tofile(fname)


        if args.data == 'rmab_context_features_lipschitz':

            B = args.budget_frac * N

            cname = "../logs/C/" + str(seedbase) + "/C.npy"
            C = np.load(cname)[:N]

            rname = "../logs/R/" + str(seedbase) + "/R.npy"
            R = np.load(rname)[:N]

            # tname = "../logs/T/" + "0" + "/T.npy"
            tname = "../logs/T/" + str(seedbase) + "/T_sinx_new.npy"
            T = np.load(tname)[:N]

            fname = "../logs/F/" + str(seedbase) + "/F_new.npy"
            F = np.load(fname)[:N]

            # qname = "../logs/Q/" + "0" + "/Q.npy"
            qname = "../logs/Q/" + str(seedbase) + "/Q_sinx_new.npy"
            Q = np.load(qname)[:N]

            vals = [[[], []], [[], []]]
            diff = [[[], []], [[], []]]
            qdiff = [[[], []], [[], []]]
            LC = np.zeros((2, 2))

            for l in range(len(Q) - 1):
                for j in range(l+1, len(Q)):
                    if abs(F[l] - F[j]) < 1e-1:
                        continue
                    for s in range(2):
                        a = 1

                        diffval = abs(F[l]-F[j])
                        qdiffval = abs(Q[l][s][a] - Q[j][s][a])
                        diff[s][a].append(diffval)
                        qdiff[s][a].append(qdiffval)
                        temp = abs(qdiffval/diffval)

                        vals[s][a].append(temp)
                        LC[s, a] = max(LC[s, a], temp)

            # vals = np.array(vals)
            # LC = np.zeros((2, 2))

            # for s in range(2):
            #     a = 1
            #     temp = np.percentile(vals[s][a], args.ptile)
            #     if args.eps != 0:
            #         newvals = []
            #         for j in range(len(diff)):
            #             t1 = qdiff[s][a][j] - args.eps
            #             t2 = diff[s][a][j]
            #             newvals.append(abs(t2/t1))
            #         temp = np.percentile(newvals, args.ptile)

            #     LC[s, a] = temp

            try:
                type_dist = []
                X = np.array(F).reshape(-1, 1)
                bandwidth = estimate_bandwidth(X, quantile=0.1)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(X)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_

                labels_unique = np.unique(labels)
                num_types = len(labels_unique)

                for k in range(num_types):
                    my_members = labels == k
                    # print("cluster {0}: {1}".format(k, X[my_members, 0]))
                    type_dist.append(len(X[my_members, 0]))

                type_dist = np.array(type_dist)
                # print(type_dist)
            except:
                num_types = 10
                type_dist = np.array([10 for _ in range(N//num_types)])

        if args.data == 'rmab_context_features':
            T, R, C, F = simulation_environments.rmab_context_features(N)
            B = args.budget_frac * N
            num_types = 10
            type_dist = np.array([10 for _ in range(N//num_types)])

        
        # Start state
        if args.start_state == -1:
            start_state = np.random.randint(args.num_states, size=N)
        elif args.start_state is not None:
            start_state = np.full(N, args.start_state)

        np.random.seed(seed=seedbase)

        T_steady = solve_steady(T, args.num_actions)

        for p, policy_option in enumerate(policies):

            policy_start_time = time.time()
            if True:
                np.random.seed(seed=seedbase)

                optimal_policy = None
                combined_state_dict = None
                if policy_option == 4:
                    optimal_policy, combined_state_dict = mdp.get_mdp_optimal_policy(
                        T, R, C, B, k, args.discount_factor)

                L_in = L

                state_matrix, action_logs, qvalues = simulateAdherence(N, L_in, T, R, C, B, k, policy_option=policy_option, seedbase=seedbase,
                                                                       action_logs=action_logs, features=F, cumulative_state_log=cumulative_state_log,
                                                                       learning_mode=LEARNING_MODE, learning_random_seed=learning_seed_base,
                                                                       world_random_seed=world_seed_base, optimal_policy=optimal_policy,
                                                                       combined_state_dict=combined_state_dict, file_root=file_root, output_data=output_data,
                                                                       start_state=start_state, do_plot=do_plot, pname=pname,
                                                                       gamma=args.discount_factor, type_dist=type_dist, LC=LC, beta=args.beta, lud=args.lud, qinit=args.qinit, eqdist=args.eqdist, nknown=args.nknown)

                np.save(file_root+'/logs/adherence_log/states_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s_start%s_lud%s_beta%s_qinit%s_eqdist%s_nknown%s' % (savestring, N,
                        args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.start_state, args.lud, args.beta, args.qinit, args.eqdist, args.nknown), state_matrix)

                qvalues_log[policy_option].append(np.array(qvalues))

                # if qvalues is not None:
                #     np.save(file_root+'/logs/qvalues/qvalues_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy' % (savestring, N,
                #         args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions), qvalues)


                # reward_matrix = np.zeros(state_matrix.shape)
                # for ind_i in range(state_matrix.shape[0]):
                #     for ind_j in range(state_matrix.shape[1]):
                #         reward_matrix[ind_i, ind_j] = (
                #             args.discount_factor**ind_j)*R[ind_i, state_matrix[ind_i, ind_j]]

                # # np.save(file_root+'/logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,trial_percent_lam0), reward_matrix)

                # state_log[policy_option].append(np.sum(reward_matrix))
                # mean_reward_log[policy_option].append(np.mean(reward_matrix.cumsum(axis=1),axis=0))
                reward_matrix = np.zeros(state_matrix.shape)

                # longterm average
                for ind_i in range(state_matrix.shape[0]):
                    for ind_j in range(state_matrix.shape[1]):
                        reward_matrix[ind_i, ind_j] = R[ind_i,
                                                        state_matrix[ind_i, ind_j]]

                if policy_option == 10:
                    for ind_i in range(len(action_logs[policy_option])):
                        for ind_j in range(len(action_logs[policy_option][ind_i])):
                            reward_matrix[ind_j, ind_i + 1] = (T_steady[ind_j][action_logs[policy_option][ind_i][ind_j]] *
                                                               R[ind_j]).sum()

                # np.save(file_root+'/logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s' % (savestring, N,
                        # args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions), reward_matrix)

                reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                reward_matrix_cumulative = reward_matrix_cumulative / \
                    (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                mean_reward_log[policy_option].append(
                    np.sum(reward_matrix_cumulative, axis=0))

                # convolved_reward_matrix = []

                # # size of convolution/sliding window
                reward_matrix_new = np.zeros(
                    (state_matrix.shape[0], window_size + 1))
                ws = window_size
                for ind_i in range(state_matrix.shape[0]):
                    # for ind_j in range(state_matrix.shape[1]):
                    # reward_matrix[ind_i, ind_j] = R[ind_i,
                    #                                 state_matrix[ind_i, ind_j]]

                    reward_matrix_new[ind_i] = np.convolve(
                        reward_matrix[ind_i], np.ones(ws)/ws, mode='valid')

                mean_reward_log_moving_avg[policy_option].append(
                    np.sum(reward_matrix_new, axis=0))

                np.set_printoptions(precision=None, threshold=np.inf, edgeitems=None,
                                    linewidth=np.inf, suppress=None, nanstr=None,
                                    infstr=None, formatter=None, sign=None,
                                    floatmode=None, legacy=None)

            policy_end_time = time.time()
            policy_run_time = policy_end_time-policy_start_time
            # np.save(file_root+'/logs/runtime/runtime_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s' % (savestring, N,
            #         args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions), policy_run_time)

            # print(N_TRIALS, len(policies))
            # print(i, p)
            runtimes[i, p] = policy_run_time

        ##### SAVE ALL RELEVANT LOGS #####

        # write out action logs
        for policy_option in action_logs.keys():
            fname = os.path.join(args.file_root, 'logs/action_logs/action_logs_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s_start%s_lud%s_beta%s_qinit%s_eqdist%s_nknown%s' % (
                savestring, N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.start_state, args.lud, args.beta, args.qinit, args.eqdist, args.nknown))
            columns = list(map(str, np.arange(N)))
            df = pd.DataFrame(action_logs[policy_option], columns=columns)
            df.to_csv(fname, index=False)

        # # write out cumulative state logs
        # for policy_option in cumulative_state_log.keys():
        #     fname = os.path.join(args.file_root,'logs/cumulative_state_log/cumulative_state_log_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s.csv'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states))
        #     columns = list(map(str, np.arange(L)))
        #     df = pd.DataFrame([cumulative_state_log[policy_option]], columns=columns)
        #     df.to_csv(fname, index=False)

        # write out T matrix logs
        # fname = os.path.join(args.file_root,'logs/Tmatrix_logs/Tmatrix_logs_'+savestring+'_N%s_k%s_L%s_data%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
        # np.save(fname, T)

    end = time.time()
    print("Time taken: ", end-start)

    for i, p in enumerate(policies):
        # print (pname[p],": ", np.mean(state_log[p]))
        print(pname[p], ": ", runtimes[:, i].mean())

    # exit()

    if True:
        # print('costs')
        # print(C)
        # print(B)
        policies_to_plot = policies


        def start_string(a):
            if a == -1:
                return "Random"
            return str(a)
        
        labels = [pname[i] for i in policies_to_plot]
        values = [np.mean(mean_reward_log_moving_avg[i], axis=0)
                  for i in policies_to_plot]

        ymin, ymax = np.array(values).min(), np.array(values).max()
        # fill_between_0 = [np.percentile(
        # mean_reward_log_moving_avg[i], 0, axis=0) for i in policies_to_plot]
        # fill_between_1 = [np.percentile(
        #     mean_reward_log_moving_avg[i], 100, axis=0) for i in policies_to_plot]
        fill_between_0 = [(j - sem(mean_reward_log_moving_avg[i], axis=0))
                          for i, j in zip(policies_to_plot, values)]
        fill_between_1 = [(j + sem(mean_reward_log_moving_avg[i], axis=0))
                          for i, j in zip(policies_to_plot, values)]

        utils.rewardPlot(labels, values, fill_between_0=fill_between_0, fill_between_1=fill_between_1,
                         ylabel='Moving average reward (ws=%s)' % window_size,
                         title='Super Arms: %s, Data: %s, Patients %s, Budget: %s, S: %s, A: %s, Start: %s, Lud %s, Beta: %s' % (
                             args.num_types, args.data, N, B, args.num_states, args.num_actions, start_string(args.start_state), args.lud, args.beta),
                        #  filename='img/online_trajectories_moving_average_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s.png' % (
                        #      savestring, N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta),
                         filename='img/online_trajectories_moving_average_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s_qinit%s_eqdist%s_nknown%s.png' % (
                             N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta, args.qinit, args.eqdist, args.nknown),
                         root=args.file_root, x_ind_for_line=-window_size)

        # bottom = 0

        labels = [pname[i] for i in policies_to_plot]
        values = [np.mean(mean_reward_log[i], axis=0)
                  for i in policies_to_plot]
        # fill_between_0 = [np.percentile(
        #     mean_reward_log[i], 25, axis=0) for i in policies_to_plot]
        # fill_between_1 = [np.percentile(
        #     mean_reward_log[i], 75, axis=0) for i in policies_to_plot]
        fill_between_0 = [(j - sem(mean_reward_log[i], axis=0))
                          for i, j in zip(policies_to_plot, values)]
        fill_between_1 = [(j + sem(mean_reward_log[i], axis=0))
                          for i, j in zip(policies_to_plot, values)]

        # errors=[np.std(np.array(state_log[i])) for i in policies_to_plot]

        # vals = [values, errors]
        # df = pd.DataFrame(vals, columns=labels)
        # fname = os.path.join(args.file_root,'logs/results/results_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s.csv'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states,trial_percent_lam0))
        # df.to_csv(fname,index=False)

        utils.rewardPlot(labels, values, fill_between_0=fill_between_0, fill_between_1=fill_between_1,
                         ylabel='Mean cumulative reward',
                         title='Super Arms: %s, Data: %s, Patients %s, Budget: %s, S: %s, A: %s, Start: %s, Lud: %s, Beta: %s' % (
                             args.num_types, args.data, N, B, args.num_states, args.num_actions, start_string(args.start_state),args.lud, args.beta),
                        #  filename='img/online_trajectories_mean_cumu_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s.png' % (
                        #      savestring, N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta),
                         filename='img/online_trajectories_mean_cumu_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s_qinit%s_eqdist%s_nknown%s.png' % (
                             N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta, args.qinit, args.eqdist, args.nknown),
                         root=args.file_root)
        
        utils.rewardPlot(labels, values, fill_between_0=fill_between_0, fill_between_1=fill_between_1,
                         ylabel='Mean cumulative reward',
                         title='Super Arms: %s, Data: %s, Patients %s, Budget: %s, S: %s, A: %s, Start: %s, Lud: %s, Beta: %s' % (
                             args.num_types, args.data, N, B, args.num_states, args.num_actions, start_string(args.start_state),args.lud, args.beta),
                        #  filename='img/online_trajectories_mean_cumu_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s.png' % (
                        #      savestring, N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta),
                         filename='img/zoomed_online_trajectories_mean_cumu_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s_qinit%s_eqdist%s_nknown%s.png' % (
                             N, args.budget_frac, L_in, policy_option, args.data, seedbase, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta, args.qinit, args.eqdist, args.nknown),
                         root=args.file_root, ylim=[ymin, ymax])
        

        opt_w = [[0.16805279, 2.12124907, 0.03011865], [0.14671956, 1.29458158, 0.03967936], [0.12072916, 0.76914539, 0.05207785]]
        gcolor = {31: 'b', 36: 'k', 61: 'c', 66:'y', 3:'m', 44:'r', 74:'g', 94: 'b'}
        plt.rcParams.update({'figure.figsize':(18,16), 'figure.dpi':100})

        '''commented out by Shahin
        # wcolor = {'r', 'g', 'b', }
        for nt in range(N_TRIALS):
            for p in policies:
                vals = np.array(qvalues_log[p][nt])
                plt.figure()
                index = 0
                for st in [0, 1, 2]:
                    stc = ['r', 'g', 'b'][st]
                    for t1 in [0, 1, 2]:
                        tmark = [',', 'v', 'o'][t1]
                        if t1 == 0:
                            l1 = 0
                            l2 = int(0.2*N)
                        elif t1 == 1:
                            l1 = int(0.2*N)
                            l2 = int(0.4*N)
                        elif t1 == 2:
                            l1 = int(0.4*N)
                            l2 = N
                        if p == 3:
                            continue
                        if p == 74:
                            r1 = t1
                            r2 = t1 + 1
                        else:
                            r1 = l1
                            r2 = l2
                        widx = np.mean(vals[:, r1:r2, st, 1] - vals[:, r1:r2, st, 0], axis=1)
                        ltt = "S=" + str(st) + " T=" + str(t1)
                        
                        plt.plot(widx, color=stc, marker=tmark, label=ltt, alpha=0.75)
                        # fill_between_0 = [(j - sem(qvalues_log[p][:, r1:r2, st, 1] - qvalues_log[p][:, r1:r2, st, 0], axis=1).reshape(-1)[k])
                        #                 for k, j in enumerate(widx)]
                        # fill_between_1 = [(j + sem(qvalues_log[p][:, r1:r2, st, 1] - qvalues_log[p][:, r1:r2, st, 0], axis=1).reshape(-1)[k])
                        #                 for k, j in enumerate(widx)]
                                        
                        # plt.fill_between(np.arange(len(widx)), fill_between_0, fill_between_1, color=mcolors.TABLEAU_COLORS[cindex], alpha=0.2)
                        # print(p, widx, fill_between_0, fill_between_1)
                        index += 1
                plt.legend()
                # sv = "../img/widx_avg/P" + str(p) + "_seed" + str(nt) + ".png"
                svf = '../img/widx_avg/P%s_N%s_b%s_L%s_data%s_seed%s_S%s_a%s_super%s_start%s_lud%s_beta%s_qinit%s_eqdist%s_nknown%s.png' % (p, N, args.budget_frac, L_in, args.data, nt, args.num_states, args.num_actions, args.num_types, args.start_state, args.lud, args.beta, args.qinit, args.eqdist, args.nknown)
                ttt = "Distribution of Whittle index for policy " + str(pname[p])
                plt.gca().set(title=ttt, xlabel="Timesteps", ylabel='Whittle Index')
                plt.savefig(svf)
                plt.close()
                plt.cla()
        '''
        
        # for nt in range(N_TRIALS):
        #     for t1 in [0, 1, 2]:
        #         if t1 == 0:
        #             l1 = 0
        #             l2 = int(0.2*N)
        #         elif t1 == 1:
        #             l1 = int(0.2*N)
        #             l2 = int(0.4*N)
        #         elif t1 == 2:
        #             l1 = int(0.4*N)
        #             l2 = N

        #         for st in [0, 1, 2]:
        #             plt.figure()
        #             for p in policies:
        #                 vals = qvalues_log[p][nt]
        #                 if p == 3:
        #                     continue
        #                 if p == 74:
        #                     r1 = t1
        #                     r2 = t1 + 1
        #                 else:
        #                     r1 = l1
        #                     r2 = l2
                        
        #                 widx = np.mean(vals[:, r1:r2, st, 1] - vals[:, r1:r2, st, 0], axis=1)
        #                 plt.plot(widx, color=gcolor[p], label=pname[p], alpha=0.75)
        #                 fill_between_0 = [(j - sem(vals[:, r1:r2, st, 1] - vals[:, r1:r2, st, 0], axis=1).reshape(-1)[k])
        #                                 for k, j in enumerate(widx)]
        #                 fill_between_1 = [(j + sem(vals[:, r1:r2, st, 1] - vals[:, r1:r2, st, 0], axis=1).reshape(-1)[k])
        #                                 for k, j in enumerate(widx)]
                                        
        #                 plt.fill_between(np.arange(len(widx)), fill_between_0, fill_between_1, color=gcolor[p], alpha=0.2)
        #                 print(p, widx, fill_between_0, fill_between_1)
        #             plt.legend()
        #             sv = "../img/widx_avg/" + str(t1) + "_W_" + str(st) + "_T" + str(t1)  + "_seed" + str(nt) + ".png"
        #             ttt = "Distribution of Whittle index for state " + str(st) + " for Arm type " + str(t1) + ", optimal: " + str(opt_w[t1][st])
        #             plt.gca().set(title=ttt, xlabel="Timesteps", ylabel='Whittle Index')
        #             plt.savefig(sv)
        #             plt.close()
        #             plt.cla()

        #             for act in [0, 1]:
        #                 plt.figure()
        #                 for p in policies:
        #                     vals = qvalues_log[p][nt]
        #                     if p == 3:
        #                         continue
        #                     if p == 74:
        #                         r1 = t1
        #                         r2 = t1 + 1
        #                     else:
        #                         r1 = l1
        #                         r2 = l2
        #                     # print(p, a1, st, act)
        #                     qv = np.mean(vals[:, r1:r2, st, act], axis=1)
        #                     plt.plot(qv, color=gcolor[p], label=pname[p], alpha=0.75)
        #                     fill_between_0 = [(j - sem(vals[:, r1:r2, st, act], axis=1).reshape(-1)[k]) for k,j in enumerate(qv)]
        #                     fill_between_1 = [(j + sem(vals[:, r1:r2, st, act], axis=1).reshape(-1)[k]) for k,j in enumerate(qv)]
        #                     plt.fill_between(np.arange(len(qv)), fill_between_0, fill_between_1, color=gcolor[p], alpha=0.2)
        #                     print(p, qv, fill_between_0, fill_between_1)

        #                 plt.legend()
        #                 sv = "../img/qvalue_graph_avg/" + str(t1) + "_Q_" + str(st) +"_" + str(act) + ".png"
        #                 ttt = "Distribution of Q(" + str(st) + "," +str(act) + ") values for Type " + str(t1) + " Opt: " + str(Q[t1][st*2 + act])
        #                 plt.gca().set(title=ttt, xlabel="Timesteps", ylabel='Q Value')
        #                 plt.savefig(sv)
        #                 plt.close()
        #                 plt.cla()

        # python simulator.py -pc -1 -d rmab_context -l 100 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 100 -lr 1 -N 20 -sts 2 -nt 4 -b 0.1 -td 40 30 20 10
        # python simulator.py -pc 10 -d rmab_context_features -l 100 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 10 -lr 1 -N 20 -sts 2 -b 0.1
        # python simulator.py -pc -1 -d arpita_circulant_dynamics -l 100 -s 0 -ws 0 -ls 0 -g 0.99 -S 4 -A 2 -n 100 -lr 1 -N 30 -sts -1 -b 0.2