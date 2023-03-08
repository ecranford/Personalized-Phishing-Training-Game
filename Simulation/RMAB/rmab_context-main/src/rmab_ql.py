import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numba import jit
import itertools
import math
import bisect


# @jit(nopython=True)
def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    return None

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    return None

BORKAR_DIVISOR = 1000

def dist(a, b):
    if type(a) is np.ndarray:
        return np.linalg.norm(a - b)
    else:
        return abs(a - b)

def rmabql_qlearn_helper(actions, state_log, R, iteration, counts,
                        ql_type, costs, Q, gamma, currentIndices, 
                        WhittleIndex):

    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        counts[arm][state][a] += 1
        alpha = 1/(counts[arm][state][a] + 1)

        a = int(a)
        # Q-Learning
        Q[arm][state][a] += alpha * \
            (R[arm, state] + gamma*max(Q[arm][nextState][:]) - Q[arm][state][a])

        if a > 0:
            if ql_type == 0:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[arm][a] - costs[arm][a-1])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[a] - costs[a-1])
            elif ql_type == 1:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][a-1])
            elif ql_type == 2:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[arm][a] - costs[arm][0])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[a] - costs[0])
            elif ql_type == 3:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][0])
        # if a == 0:
        #   self.WhittleIndex[arm, state, a] = self.Q[arm][state][a]

        # Create a list of current Whittle Indices based on each arm's current state
        currentIndices[arm] = WhittleIndex[arm, nextState]
        # print(currentIndices)
        # WhittleIndexOverTime[arm, it    eration] = WhittleIndex[arm]


def rmabql_qlearn_helper_lipschitz(actions, state_log, R, iteration, counts, ql_type,
                                   costs, Q, gamma, currentIndices, WhittleIndex, F, LC,
                                   K, beta, eqdist, nknown):

    approxQ = np.full(Q.shape, 10000.0)
    dflag = np.full(Q.shape, 10000.0)

    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        counts[arm][state][a] += 1
        alpha = 1/(counts[arm][state][a] + 1)

        a = int(a)
        # Q-Learning
        Q[arm][state][a] += alpha * \
            (R[arm, state] + gamma*max(Q[arm][nextState][:]) - Q[arm][state][a])

        if a > 0:

            if ql_type == 0:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[arm][a] - costs[arm][a-1])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[a] - costs[a-1])
            elif ql_type == 1:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][a-1])
            elif ql_type == 2:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[arm][a] - costs[arm][0])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[a] - costs[0])
            elif ql_type == 3:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][0])

            if counts[arm][state][a] > nknown:
                for arm2, a2 in enumerate(actions):
                    if a2 == 0:
                        distance = dist(F[arm], F[arm2])
                        lupdate = Q[arm, state, a] + (LC[state, a] * distance) + K
                        minval = min(approxQ[arm2, state, a], lupdate)
                        if lupdate < approxQ[arm2, state, a]:
                            approxQ[arm2, state, a] = minval
                            dflag[arm2, state, 1] = distance

        # if a == 0:
        #   self.WhittleIndex[arm, state, a] = self.Q[arm][state][a]
    # print()
    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        a = int(a)
        if a == 0:
            for s in range(int(Q.shape[1])):
                if approxQ[arm, s, 1] != 10000:
                    if dflag[arm, s, 1] == 0 and eqdist:
                        counts[arm][s][1] += 1
                    beta = 1/(counts[arm][s][1] + 1)
                    # if arm == 16:
                    #     print(beta, 1-beta, Q[arm, s, 1], approxQ[arm, s, 1])
                    Q[arm, s, 1] = (1-beta) * Q[arm, s, 1] + beta*approxQ[arm, s, 1]
                    WhittleIndex[arm, s, 1] = (Q[arm, s, 1] - Q[arm, s, 0])
        
        # Create a list of current Whittle Indices based on each arm's current state
        currentIndices[arm] = WhittleIndex[arm, nextState]
        

def rmabql_qlearn_helper_dabel_lipschitz(actions, state_log, R, iteration, counts, ql_type,
                                   costs, Q, gamma, currentIndices, WhittleIndex, F, LC,
                                   eqdist, nknown, T):

    approxQ = np.full(Q.shape, 10000.0)
    dflag = np.full(Q.shape, 10000.0)

    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        counts[arm][state][a] += 1
        alpha = 1/(counts[arm][state][a] + 1)

        a = int(a)
        # Q-Learning
        Q[arm][state][a] += alpha * \
            (R[arm, state] + gamma*max(Q[arm][nextState][:]) - Q[arm][state][a])

        if a > 0:

            if ql_type == 0:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[arm][a] - costs[arm][a-1])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[a] - costs[a-1])
            elif ql_type == 1:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][a-1])
            elif ql_type == 2:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[arm][a] - costs[arm][0])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[a] - costs[0])
            elif ql_type == 3:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][0])

            if counts[arm][state][a] > nknown:
                for arm2, a2 in enumerate(actions):
                    if a2 == 0:
                        distance = dist(F[arm], F[arm2])
                        t1 = 0
                        t2 = 0
                        for s1 in range(Q.shape[1]):
                            temp = max(Q[arm2, s1])
                            temp2 = max(Q[arm, s1])
                            t1 += temp
                            t2 += T[arm, state, a, s1] * abs(temp - temp2)
                        
                        lupdate = Q[arm, state, a] + gamma * (LC[state, a] * distance * t1) + gamma * t2
                        minval = min(approxQ[arm2, state, a], lupdate)
                        if lupdate < approxQ[arm2, state, a]:
                            approxQ[arm2, state, a] = minval
                            dflag[arm2, state, 1] = distance

        # if a == 0:
        #   self.WhittleIndex[arm, state, a] = self.Q[arm][state][a]
    # print()
    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        a = int(a)
        if a == 0:
            for s in range(int(Q.shape[1])):
                if approxQ[arm, s, 1] != 10000:
                    if dflag[arm, s, 1] == 0 and eqdist:
                        counts[arm][s][1] += 1
                    beta = 1/(counts[arm][s][1] + 1)
                    # if arm == 16:
                    #     print(beta, 1-beta, Q[arm, s, 1], approxQ[arm, s, 1])
                    Q[arm, s, 1] = (1-beta) * Q[arm, s, 1] + beta*approxQ[arm, s, 1]
                    WhittleIndex[arm, s, 1] = (Q[arm, s, 1] - Q[arm, s, 0])
        
        # Create a list of current Whittle Indices based on each arm's current state
        currentIndices[arm] = WhittleIndex[arm, nextState]
        

def rmabql_qlearn_helper_lipschitz_optimisticQ(actions, state_log, R, iteration, counts, ql_type,
                                   costs, Q, gamma, currentIndices, WhittleIndex, F, LC,
                                   K, beta, M, eqdist, nknown):
    approxQ = np.full(Q.shape, 10000.0)
    acted = dict([(key, []) for key in range(Q.shape[1])])

    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        counts[arm][state][a] += 1
        alpha = 1/(counts[arm][state][a] + 1)

        a = int(a)
        # Q-Learning
        Q[arm][state][a] += alpha * \
            (R[arm, state] + gamma*max(Q[arm][nextState][:]) - Q[arm][state][a])

        if a > 0:
            acted[state].append(arm)
            if ql_type == 0:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[arm][a] - costs[arm][a-1])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[a] - costs[a-1])
            elif ql_type == 1:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][a-1])
            elif ql_type == 2:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[arm][a] - costs[arm][0])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[a] - costs[0])
            elif ql_type == 3:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][0])

            # for arm2, a2 in enumerate(actions):
            #     if a2 == 0:
            #         lupdate = Q[arm, state, a] + (LC[state, a] * dist(F[arm], F[arm2]) + K)/((counts[arm][state][a] + 1)**M)
            #         minval = min(approxQ[arm2, state, a], lupdate)
            #         if lupdate < approxQ[arm2, state, a]:
            #             approxQ[arm2, state, a] = minval

        # if a == 0:
        #   self.WhittleIndex[arm, state, a] = self.Q[arm][state][a]
    # print()
    for arm, a in enumerate(actions):
        s = state_log[arm, 0]
        nextState = state_log[arm, 1]

        a = int(a)
        if a == 0:
            for s in range(int(Q.shape[1])):
                minval = 10000.0
                dflag = False
                for act_arm in acted[s]:
                    if counts[act_arm][s][1] <= nknown:
                        continue
                    distance = dist(F[act_arm], F[arm])

                    lupdate = Q[act_arm, s, 1] + (LC[s, 1] * distance + K)/((counts[act_arm][s][1] + 1)**M)
                    if lupdate < minval:
                        minval = lupdate
                        if distance == 0:
                            dflag = True
                        else:
                            dflag = False

                if minval < 10000.0:
                    if dflag and eqdist:
                        counts[arm][state][1] += 1
                    beta = 1/(counts[arm][s][1] + 1)
                    Q[arm, s, 1] = (1-beta) * Q[arm, s, 1] + beta*minval
                    WhittleIndex[arm, s, 1] = (Q[arm, s, 1] - Q[arm, s, 0])


            # for state in range(Q.shape[1]):
            #     minval = 10000.0
            #     left_idx = find_lt(acted[state], arm)
            #     right_idx = find_gt(acted[state], arm)
            #     if left_idx is not None:
            #         temp = Q[left_idx, state, 1] + (LC[state, 1] * dist(F[arm], F[left_idx]) + K)/((counts[left_idx][state][1]+ 1)**M)
            #         if temp < minval:
            #             minval = temp
            #     if right_idx is not None:
            #         temp = Q[right_idx, state, 1] + (LC[state, 1] * dist(F[arm], F[right_idx]) + K)/((counts[right_idx][state][1]+ 1)**M)
            #         if temp < minval:
            #             minval = temp
                
            #     if minval != 10000.0:
            #         beta = 1/(counts[arm][state][1] + 1)
            #         Q[arm, state, 1] = (1-beta) * Q[arm, state, 1] + beta * minval
            #         WhittleIndex[arm, state, 1] = (Q[arm, state, 1] - Q[arm, state, 0])


        
        # Create a list of current Whittle Indices based on each arm's current state
        currentIndices[arm] = WhittleIndex[arm, nextState]
        # print(currentIndices)
        # WhittleIndexOverTime[arm, iteration] = WhittleIndex[arm]


def rmabql_qlearn_helper_lipschitz_optimisticQ_add(actions, state_log, R, iteration, counts, ql_type,
                                   costs, Q, gamma, currentIndices, WhittleIndex, F, LC,
                                   K, beta, M, eqdist, nknown):
    approxQ = np.full(Q.shape, 10000.0)
    acted = dict([(key, []) for key in range(Q.shape[1])])

    for arm, a in enumerate(actions):
        state = state_log[arm, 0]
        nextState = state_log[arm, 1]

        counts[arm][state][a] += 1
        alpha = 1/(counts[arm][state][a] + 1)

        a = int(a)
        # Q-Learning
        Q[arm][state][a] += alpha * \
            (R[arm, state] + gamma*max(Q[arm][nextState][:]) - Q[arm][state][a])

        if a > 0:
            acted[state].append(arm)
            if ql_type == 0:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[arm][a] - costs[arm][a-1])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][a-1])/(costs[a] - costs[a-1])
            elif ql_type == 1:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][a-1])
            elif ql_type == 2:
                try:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[arm][a] - costs[arm][0])
                except:
                    WhittleIndex[arm, state, a] = (
                        Q[arm][state][a] - Q[arm][state][0])/(costs[a] - costs[0])
            elif ql_type == 3:
                WhittleIndex[arm, state, a] = (
                    Q[arm][state][a] - Q[arm][state][0])

            # for arm2, a2 in enumerate(actions):
            #     if a2 == 0:
            #         lupdate = Q[arm, state, a] + (LC[state, a] * dist(F[arm], F[arm2]) + K)/((counts[arm][state][a] + 1)**M)
            #         minval = min(approxQ[arm2, state, a], lupdate)
            #         if lupdate < approxQ[arm2, state, a]:
            #             approxQ[arm2, state, a] = minval

        # if a == 0:
        #   self.WhittleIndex[arm, state, a] = self.Q[arm][state][a]
    # print()
    for arm, a in enumerate(actions):
        s = state_log[arm, 0]
        nextState = state_log[arm, 1]

        a = int(a)
        if a == 0:
            for s in range(int(Q.shape[1])):
                minval = 10000.0
                dflag = False
                for act_arm in acted[s]:
                    distance = dist(F[act_arm], F[arm])

                    lupdate = Q[act_arm, s, 1] + (LC[s, 1] * distance) + 1/((counts[act_arm][s][1] + 1)**M)
                    if lupdate < minval:
                        minval = lupdate
                        if distance == 0:
                            dflag = True
                        else:
                            dflag = False

                if minval < 10000.0:
                    if dflag and eqdist:
                        counts[arm][state][1] += 1
                    beta = 1/(counts[arm][s][1] + 1)
                    Q[arm, s, 1] = (1-beta) * Q[arm, s, 1] + beta*minval
                    WhittleIndex[arm, s, 1] = (Q[arm, s, 1] - Q[arm, s, 0])
        
        # Create a list of current Whittle Indices based on each arm's current state
        currentIndices[arm] = WhittleIndex[arm, nextState]


class RMABQL(object):
    def __init__(self, n_arms, m, eps, alpha, gamma,
                 iterations, n_states, n_actions, initial_exploration=False,
                 eps_decay=False, ql_type=None,
                 qinit=0, eqdist=False, nknown=0):

        self.n_arms = n_arms  # Number of arms
        self.eps = eps  # USed for the \epsilon-greedy selection of arms
        self.alpha = alpha  # Learning parameter for Q values
        self.gamma = gamma  # Learning parameters for WhittleIndices
        # Number of iterations for simulating the Bandits environment
        self.iterations = iterations

        self.s = n_states
        self.a = n_actions

        if qinit != 0:
            self.Q = np.full((n_arms, self.s, self.a), qinit)
        else:
            self.Q = np.zeros((n_arms, self.s, self.a))  # Stores Q values

        self.eqdist = eqdist                # If equal distance, update learning rate - Flag for that
        self.nknown = nknown                # No. of (s,a) visits before using that as for lipschitz transfer

        BIG_CONST = 1000000

        # Stores the number of times each (arm, state, action) pair is observed till time t
        self.counts = np.zeros((n_arms, self.s, self.a))
        # Stores the values of Whittle Indices (of an arm at each state), which is learnt till time t
        self.WhittleIndex = np.zeros((n_arms, self.s, self.a))
        # Stores the values of Whittle Indices (of an arm at each state), which is learnt till time t
        self.WhittleIndexOverTime = np.zeros(
            (n_arms, self.iterations, self.s, self.a))
        # Stores the current values of Whittle Indices of the currentState for each arm.
        self.currentIndices = np.zeros((n_arms, self.a))*(-BIG_CONST)
        # Stores the number of times a state is pulled observed till time t (cumulative)
        self.count_state = np.zeros(((self.iterations+1), self.s))

        self.initial_exploration = initial_exploration
        self.eps_decay = eps_decay
        self.ql_type = ql_type

    def epsilon(self, time):
        A = 0.4
        B = 0.2
        C = 0.1
        standardized_time=(time-A*self.iterations)/(B*self.iterations)
        cosh=np.cosh(math.exp(-standardized_time))
        epsilon=1.1-(1/cosh+(time*C/self.iterations))
        return epsilon

    def check_random(self, iteration, random_stream=None):
        # eps = 1
        if self.eps_decay:
            # self.eps = 1/(1+(iteration/(self.n_arms+self.s)))
            # eps = max(self.eps/np.ceil(iteration/BORKAR_DIVISOR), 0)
            eps = self.epsilon(iteration)
            # eps = self.n_arms/(self.n_arms + iteration)
        # else:
            

        # if self.initial_exploration:
        #     if self.iterations < 100:
        #         self.eps = 0.9

        p = None
        if random_stream is not None:
            p = random_stream.random()
        else:
            p = np.random.random()

        return p < eps

    # need to return nxa array of indices

    def get_indexes(self):
        return self.currentIndices

    def get_whittleindexes(self):
        return self.WhittleIndex

    # action is dimension n
    # state_log is nx2 array with prev state and current state
    # R is nxs

    def qlearn(self, actions, state_log, R, iteration, costs):

        # Take actions based on the selection
        for arm, a in enumerate(actions):
            if a > 0:
                # update the state counts only when you act but not sure why
                self.count_state[iteration][state_log[arm][0]] += 1

        for state in range(self.s):
            self.count_state[iteration +
                             1][state] = self.count_state[iteration][state]

        # Update values (for selecting next m arms)
        rmabql_qlearn_helper(actions, state_log, R, iteration, self.counts, self.ql_type,
                             costs, self.Q, self.gamma, self.currentIndices, self.WhittleIndex)

    def qlearn_dabel_lipschitz(self, actions, state_log, R, iteration, costs, F, LC, T):

        # Take actions based on the selection
        for arm, a in enumerate(actions):
            if a > 0:
                # update the state counts only when you act but not sure why
                self.count_state[iteration][state_log[arm][0]] += 1

        for state in range(self.s):
            self.count_state[iteration +
                             1][state] = self.count_state[iteration][state]

        # Update values (for selecting next m arms)
        rmabql_qlearn_helper_dabel_lipschitz(actions, state_log, R, iteration, self.counts, self.ql_type,
                             costs, self.Q, self.gamma, self.currentIndices, self.WhittleIndex, F, LC,
                             self.eqdist, self.nknown, T)

    def qlearn_lipschitz(self, actions, state_log, R, iteration, costs, F, LC, K, beta):

        # Take actions based on the selection
        for arm, a in enumerate(actions):
            if a > 0:
                # update the state counts only when you act but not sure why
                self.count_state[iteration][state_log[arm][0]] += 1

        for state in range(self.s):
            self.count_state[iteration +
                             1][state] = self.count_state[iteration][state]

        # Update values (for selecting next m arms)
        rmabql_qlearn_helper_lipschitz(actions, state_log, R, iteration, self.counts, self.ql_type,
                                       costs, self.Q, self.gamma, self.currentIndices, self.WhittleIndex, F, LC, K, beta,
                                       self.eqdist, self.nknown)
    
    def qlearn_lipschitz_optimistic(self, actions, state_log, R, iteration, costs, F, LC, K, beta, M=1):

        # Take actions based on the selection
        for arm, a in enumerate(actions):
            if a > 0:
                # update the state counts only when you act but not sure why
                self.count_state[iteration][state_log[arm][0]] += 1

        for state in range(self.s):
            self.count_state[iteration +
                             1][state] = self.count_state[iteration][state]

        # Update values (for selecting next m arms)
        rmabql_qlearn_helper_lipschitz_optimisticQ(actions, state_log, R, iteration, self.counts, self.ql_type,
                                       costs, self.Q, self.gamma, self.currentIndices, self.WhittleIndex, F, LC, K, beta, M,
                                       self.eqdist, self.nknown)
    
    def qlearn_lipschitz_optimistic_add(self, actions, state_log, R, iteration, costs, F, LC, K, beta, M=1):

        # Take actions based on the selection
        for arm, a in enumerate(actions):
            if a > 0:
                # update the state counts only when you act but not sure why
                self.count_state[iteration][state_log[arm][0]] += 1

        for state in range(self.s):
            self.count_state[iteration +
                             1][state] = self.count_state[iteration][state]

        # Update values (for selecting next m arms)
        rmabql_qlearn_helper_lipschitz_optimisticQ_add(actions, state_log, R, iteration, self.counts, self.ql_type,
                                       costs, self.Q, self.gamma, self.currentIndices, self.WhittleIndex, F, LC, K, beta, M,
                                       self.eqdist, self.nknown)

    def qlearn_super(self, actions, state_log, R, type_dist):
        i = 0
        for sarm, sarm_size in enumerate(type_dist):
            for arm in range(i, i + sarm_size):
                state = int(state_log[arm, 0])
                nextState = int(state_log[arm, 1])
                a = int(actions[arm])
                self.counts[sarm][state][a] += 1
                alpha = 1/(self.counts[sarm][state][a] + 1)

                self.Q[sarm][state][a] += alpha * \
                            (R[arm, state] + self.gamma * max(self.Q[sarm][nextState][:]) - self.Q[sarm][state][a])
            
            for state in range(self.s):
                for a in range(self.a):
                    self.WhittleIndex[sarm, state, a] = (self.Q[sarm][state][a] - self.Q[sarm][state][0])
            
            i += sarm_size

    def getAllIndexes(self, costs):

        for arm in range(self.WhittleIndex.shape[0]):
            for state in range(self.WhittleIndex.shape[1]):
                for a in range(1, self.WhittleIndex.shape[2]):
                    self.WhittleIndex[arm, state, a] = (
                        self.Q[arm][state][a] - self.Q[arm][state][0])/(costs[a] - costs[0])

        return self.WhittleIndex

    def plot_indexes(self):
        import matplotlib.pyplot as plt
        SMALL_SIZE = 12
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 16
        plt.rc('font', weight='bold')
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=BIGGER_SIZE)
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        fig, ax = plt.subplots(2, 3, figsize=(14, 8))
        ax = ax.reshape(-1)
        colors = ['r', 'g', 'b', 'c']
        wi_vals = [-0.5, 0.5, 1, -1]
        for arm in range(self.n_arms):
            for state in range(self.s):
                if arm == 0:
                    ax[arm].plot(self.WhittleIndexOverTime[arm, :, state, 1],
                                 color=colors[state], alpha=0.5, label='S:%s' % state)
                else:
                    ax[arm].plot(self.WhittleIndexOverTime[arm, :,
                                 state, 1], color=colors[state], alpha=0.5)

                ax[arm].plot([0, self.iterations], [
                             wi_vals[state], wi_vals[state]], color=colors[state], linestyle='--')
        fig.suptitle('Whittle Index Vals')
        fig.legend(ncol=4, loc='lower center')
        plt.savefig('indices_over_time_arm%s_wi.png' % arm, dpi=200)
        plt.show()

        fig, ax = plt.subplots(2, 3, figsize=(14, 8))
        ax = ax.reshape(-1)
        colors = ['r', 'g', 'b', 'c']
        wi_vals = [-0.9047619,  0.9047619,  0.9047619, -0.9047619]
        for arm in range(self.n_arms):
            for state in range(self.s):
                if arm == 0:
                    ax[arm].plot(self.WhittleIndexOverTime[arm, :, state, 1],
                                 color=colors[state], alpha=0.5, label='S:%s' % state)
                else:
                    ax[arm].plot(self.WhittleIndexOverTime[arm, :,
                                 state, 1], color=colors[state], alpha=0.5)

                ax[arm].plot([0, self.iterations], [
                             wi_vals[state], wi_vals[state]], color=colors[state], linestyle='--')
        plt.suptitle('VfNc Index Vals')
        fig.legend(ncol=4, loc='lower center')
        plt.savefig('indices_over_time_arm%s_vfnc.png' % arm, dpi=200)
        plt.show()

    def showAvgRewards(self, cumulative=False, printDetails=False):
        '''
        Computes the average reward per time t
        - if cumulative = True, the cumulative average reward is computed
        - if cumulative = False, the per time total reward is computed
        '''
        avgReward = np.zeros(self.iterations)
        for i in np.arange(self.n_arms):
            if printDetails:
                print("Arm {}:".format(i))
                self.arms[i].showHistory()
                print()
            avgReward += self.arms[i].history_rewards

        if cumulative:
            for t in np.arange(len(avgReward))[1:]:
                avgReward[t] = (avgReward[t-1]*(t) + avgReward[t])/(t+1)

        if printDetails:
            print("Average reward: {}".format(avgReward))

        return avgReward

    def getQvalues(self):
        return self.Q
