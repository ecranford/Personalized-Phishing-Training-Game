#from email import policy
import numpy as np
#import pandas as pd
import time
import utils
import itertools
from itertools import product
#from scipy.stats import sem
#from sklearn.cluster import KMeans
#from sklearn.cluster import MeanShift, estimate_bandwidth
import lp_methods
#import mdp
import os
import argparse
#import tqdm
#import itertools
#import mdptoolbox
import rmab_ql
import simulation_environments
#import matplotlib.pyplot as plt
#import sys
#import random
####CODE FOR PYTHON TCP COMMUNICATION###
import json
import socketserver

##Command line to start task
#python simulator.py -pc 44 -d rmab_cog_sim -l 60 -s 0 -ws 0 -ls 0 -g 0.99 -S 2 -A 2 -n 10 -lr 1 -N 1 -sts -1 -b 0.2

HOST = "127.0.0.1"
PORT = 9143

TS_policies = [21, 22, 25, 101, 102]

parser = argparse.ArgumentParser(description='Run adherence simulations with various methods.')
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
                                                            'arpita_healthcare_static', 'arpita_circulant_dynamics_prewards', 'simple_spam_ham', 'simple_spam_ham_2', 'rmab_cog_sim'],
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

file_root = args.file_root

# Save special name
if args.save_string == '':
    args.save_string = str(
        time.ctime().replace(' ', '_').replace(':', '_'))

# Policies to run
if args.policy < 0:
    policies = [0, 3, 74]
else:
    policies = [args.policy]

# policy names dict
pname = {
    3: 'FastRandom',
    44: 'WIQL4',
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

##Set up dictionary for GroupData objects
group_data = {}
group_num = 0
##Set up dictionary for QL ojbects
qlearning_objects = {}

do_plot = False

T = None
R = None
C = None
B = None
start_state = None

## --------------------------------
##Define number of groups and players and transition probabilities, etc.
if args.data == 'rmab_cog_sim':
    T, R, C, F = simulation_environments.rmab_cog_sim(N)
    B = args.budget_frac * N
    LC = np.zeros((T.shape[1], T.shape[2]))
    num_types = 10 ## is this 1 or 10 if no clusters?? May not even be needed
    type_dist = [N]
    args.num_states = T.shape[1]

if args.data == 'simple_spam_ham':
    T, R, C, F = simulation_environments.simple_spam_ham(N)
    B = args.budget_frac * N
    LC = np.zeros((T.shape[1], T.shape[2]))
    num_types = 4
    type_dist = [int(0.21 * N), int(0.214*N), int(0.305*N), int(0.271*N)]
    args.num_states = T.shape[1]
## --------------------------------

#qlearning variables
eps = 0.75
gamma = args.discount_factor
alpha = 0.4
n_states = T.shape[1]
n_actions = T.shape[2]
ql_type = 3 #will be 3 for all policies greater than 40 and ending in 4 (we will use 44)
qinit = args.qinit
eqdist = args.eqdist

if qinit != 0:
    qinit = np.max(R)/(1-gamma)
if eqdist != 0:
    eqdist = True
else:
    eqdist = False

nknown=args.nknown

# Start state
if args.start_state == -1:
    start_state = np.random.randint(args.num_states, size=N)
elif args.start_state is not None:
    start_state = np.full(N, args.start_state)

##Do I need to reseed the random seedbase here
##np.random.seed(seed=args.seed_base)

## Function to estimate next states for QLearning
## If using cognitive model, this will need to change so that all next_states are what is observed by the model (i.e., might not need call takeAction at all in that case)
def takeAction(current_states, T, actions, observed_states, random_stream, Cog_States=False):

    N = len(current_states)

    # Get next state
    # print(current_states)
    # print(T.shape)
    next_states = np.zeros(current_states.shape)
    
    for i in range(N):

        current_state = int(current_states[i])
        if Cog_States:
            next_states[i] = observed_states[i]
        elif(int(actions[i])==1):
            next_states[i] = observed_states[i]
        else:
            next_state = np.argmax(random_stream.multinomial(1, T[i, current_state, int(actions[i]), :]))
            next_states[i] = next_state

        # if current_state != next_state:
        #     print(i, current_state, int(actions[i]), next_state, T[i, current_state, int(actions[i]), :])

    # print(next_states)

    return next_states


def getActions(N, T_hat, R, C, B, k, features=None, seed=None, valid_action_combinations=None, combined_state_dict=None, current_state=None,
               optimal_policy=None, policy_option=0, gamma=0.95, indexes=None, type_dist=None,
               output_data=None, True_T=None, qlearning_objects_group=None, learning_random_stream=None, t=None,
               action_to_play=None, current_state_log=None):
    '''
    print("N is", N)
    print("T_hat is", T_hat)
    print("R is", R)
    print("C is", C)
    print("B is", B)
    print("k is", k)
    print("features is", features)
    print("seed is", seed)
    print("valid_action_combinations is", valid_action_combinations)
    print("combined_state_dict is", combined_state_dict)
    print("current_state is", current_state)
    print("optimal_policy is", optimal_policy)
    print("policy_option is", policy_option)
    print("gamma is", gamma)
    print("indexes is", indexes)
    print("type_dist is", type_dist)
    print("output_data is", output_data)
    print("True_T is", True_T)
    print("qlearning_objects_group is", qlearning_objects_group)
    print("learning_random_stream is", learning_random_stream)
    print("t is", t)
    print("action_to_play is", action_to_play)
    print("current_state_log is", current_state_log)
    '''

    #declare globals
    global qlearning_objects

    # Fast random, inverse weighted
    if policy_option == 3:

        actions = np.zeros(N, dtype=int)
        #print("actions is", actions)

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

        #print("actions is", actions)
        return actions

    # Q-learning
    elif policy_option == 44:
        actions = np.zeros(N, dtype=int)
        #print("actions is", actions)

        print("QL object is", qlearning_objects[qlearning_objects_group])
        # with prob epsilon, explore randomly
        # This call will also decay epsilon
        if qlearning_objects[qlearning_objects_group].check_random(t+1, random_stream=learning_random_stream):
            print('Doing a random')
            if N <= 10:
                return getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                  policy_option=3, combined_state_dict=combined_state_dict,
                                  indexes=indexes, output_data=output_data, True_T=True_T,
                                  t=t, qlearning_objects_group=qlearning_objects_group)
            else:
                return getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                                  policy_option=3, combined_state_dict=combined_state_dict,
                                  indexes=indexes, output_data=output_data, True_T=True_T,
                                  qlearning_objects_group=qlearning_objects_group)

        print('Exploiting')

        # # otherwise act greedily
        indexes_per_state = qlearning_objects[qlearning_objects_group].get_indexes()
        print("indexes is", indexes_per_state)
        # indexes_per_state = indexes_per_state.cumsum(axis=1)

        # print('indexes')
        # print(indexes_per_state)
        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)
        # print("knapsack time:",time.time() - start)
        print("decision_matrix is", decision_matrix)

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
        
        #print("actions is", actions)
        return actions

def processRequest(groupID,
                   trial,
                   observed_states,
                   Cog_States=False):
    ##declare globals
    global group_data
    global qlearning_objects
    
    ##if trial > 0, takeAction and update qlearning_objects
    if trial > 0:
        ##actions and state log will be actions and state log from previous trial
        #print("State log is", group_data[groupID].state_log[:, trial-1])
        #print("Action log is", group_data[groupID].action_logs[trial-1])
        group_data[groupID].state_log[:, trial] = takeAction(group_data[groupID].state_log[:, trial-1].reshape(-1), 
                                                            T, 
                                                            group_data[groupID].action_logs[trial-1], 
                                                            observed_states, 
                                                            random_stream=group_data[groupID].world_random_stream,
                                                            Cog_States=Cog_States)

        #Update q_learning
        qlearning_objects[groupID].qlearn(group_data[groupID].action_logs[trial-1], 
                                          group_data[groupID].state_log[:, trial-1:], 
                                          R, 
                                          trial, 
                                          C)
        qvalues = qlearning_objects[groupID].getQvalues()
        group_data[groupID].qvalues_log.append(qvalues.copy())
        
        print("Updated Qvalues is ", qvalues)
        
        #Update counts
        if LEARNING_MODE == 1:
            group_data[groupID].update_counts(group_data[groupID].action_logs[trial-1], group_data[groupID].state_log[:, trial-1:])
    
    ##Setup variables for current trial before GetAction() runs
    T_hat = None

    ##Then run GetAction() to get what email types to send players (and store data to requisite objects)
    actions = getActions(N, T_hat, R, C, B, k, 
                         valid_action_combinations=group_data[groupID].valid_action_combinations, 
                         current_state=group_data[groupID].state_log[:, trial],
                         optimal_policy=group_data[groupID].optimal_policy,
                         type_dist=type_dist,
                         policy_option=group_data[groupID].policy_option, 
                         combined_state_dict=group_data[groupID].combined_state_dict, 
                         gamma=gamma,
                         indexes=group_data[groupID].indexes,
                         output_data=output_data, 
                         True_T=T, 
                         learning_random_stream=group_data[groupID].learning_random_stream,
                         t=trial,
                         qlearning_objects_group=groupID)
    print("Returned Actions Vector is:", actions)
    
    group_data[groupID].actions_record[:, trial] = actions
    #print("Actions record is:", group_data[groupID].actions_record)

    ##Update action_logs
    if group_data[groupID].action_logs is not None:
        group_data[groupID].action_logs.append(actions.astype(int))
    #print("Action_logs is:", group_data[groupID].action_logs)

    ##Then return data object that contains state of RMAB (and all sub-data objects therein) and actions
    return actions

##Define a class to store group specific data and states, identified by group id
class GroupData(object):
    def __init__(self, groupID, policy_option, observed_states, Cog_States=False):

        self.groupID = groupID
        self.policy_option = policy_option
        self.state_log = {}
        self.action_logs = {}
        self.cumulative_state_log = {}
        self.combined_state_dict = None
        self.qvalues_log = {}
        self.start = time.time()

        self.optimal_policy = None
        self.opt_act = np.zeros((L, 100, 3))
        self.tot_act = np.zeros((L, 100, 3))
        self.qv = np.frompyfunc(list, 0, 1)(np.empty((L,100, args.num_states, args.num_actions, 3), dtype=object))
        self.dqv = np.frompyfunc(list, 0, 1)(np.empty((L,100, args.num_states, 3), dtype=object))
        
        # use np global seed for rolling random data, then for random algorithmic choices
        self.seedbase = first_seedbase + group_num
        np.random.seed(seed=self.seedbase) ##do I want to reseed here?
        # Use world seed only for evolving the world (If two algs
        # make the same choices, should create the same world for same seed)
        self.world_seed_base = first_world_seedbase + group_num
        # Use learning seed only for processes involving learning (i.e., exploration vs. exploitation)
        self.learning_seed_base = first_learning_seedbase + group_num
        print("# Seed is", self.seedbase)
        ##Random states...do i need to do this once at start and save to Simulation/Nodegame data?
        self.learning_random_stream = np.random.RandomState()
        if LEARNING_MODE > 0:
            self.learning_random_stream.seed(self.learning_seed_base)
        self.world_random_stream = np.random.RandomState()
        self.world_random_stream.seed(self.world_seed_base)


        self.T_hat = None
        self.priors = np.ones(T.shape) ##This one is never updated
        self.counts = np.zeros(T.shape) ##But this one is updated after each trial

        self.qvalues = None
        self.qvalues_log = []

        self.state_log = np.zeros((N, L), dtype=int)
        self.actions_record = np.zeros((N, L)) ##THIS IS USUALLY L-1 BUT IT IS NOT LONG ENOUGH TO DO 60 TRIALS, DOES GAME NOT GO TO 60TH TRIAL???
        
        self.action_logs = []

        self.indexes = np.zeros((N, C.shape[0]))

        # start state will be -1 set by params, so state_log[:, 0] will be an array of random states of length N
        if Cog_States:
            self.state_log[:, 0] = observed_states
        elif start_state is not None:
            self.state_log[:, 0] = start_state
        else:
            self.state_log[:, 0] = 1
        
        #if problem size is small enough, enumerate all valid actions to use for random exploration, else we will use "fast" random which has some issues right now
        #N will be 10 so should just be None
        self.valid_action_combinations = None
        if policy_option in [2, 41, 42, 43, 44] and N <= 5:
            options = np.array(list(product(np.arange(C.shape[0]), repeat=N)))
            self.valid_action_combinations = utils.list_valid_action_combinations(
                N, C, B, options)

    ##update counts inside GroupData class to keep it local to group
    def update_counts(self, actions, state_log):
        for arm, a in enumerate(actions):
            a = int(a)
            s = state_log[arm, 0]
            sprime = state_log[arm, 1]
            self.counts[arm, s, a, sprime] += 1


class ExampleHandler (socketserver.StreamRequestHandler):

    def handle(self):
        #declare globals
        global group_data
        global group_num
        global qlearning_objects
        global N

        user_data = self.rfile.readline().decode("utf-8").strip()
        print(f"# received message '{user_data}'")
        user_data = json.loads(user_data)
        print(f"# decoded JSON {user_data}")
        groupID = user_data['GROUP-ID'] 
        user_list = list(user_data['OBS-STATES'].keys())
        observed_states = np.full(len(user_list), list(user_data['OBS-STATES'].values())) 
        print("# observed states are", observed_states)
        trial = user_data['TRIAL']
        if user_data['COG-STATES'] == 1:
            Cog_States = True
        else:
            Cog_States = False
        ## if GroupData object already created for group (i.e., this is not the first trial),
        ## then access GroupData object and process requeset for that group,
        ## else set up GroupData object (and rmab_ql object?)
        if groupID in group_data:
            ##Process request
            print(f"# Processing request for existing group: {groupID} trial: {trial}")
            selected_users = processRequest(groupID,trial,observed_states, Cog_States=Cog_States)
        else:
            ## set up group specific data object to save states
            group_data[groupID] = GroupData(groupID, policies[0], observed_states, Cog_States=Cog_States)
            
            #print("# New group num is:", group_num)
            #print("# State log is", group_data[groupID].state_log)
            #print("# Action logs is", group_data[groupID].action_logs)
            group_num = group_num + 1
            #print("# Next group num is:", group_num)

            ## set up QL_object
            qlearning_objects[groupID] = rmab_ql.RMABQL(N, k, eps, alpha, gamma,
                                                        L, n_states, n_actions, initial_exploration=False,
                                                        eps_decay=True, ql_type=ql_type,
                                                        qinit=qinit, eqdist=eqdist, nknown=nknown)
            ##Process request
            print(f"# Processing request for new group: {groupID} trial: {trial}")
            selected_users = processRequest(groupID,trial,observed_states, Cog_States=Cog_States)

        #print("user list is", user_list)
        #print("selected users is", selected_users)
        selected_ids = list(itertools.compress(user_list,selected_users))
        print(f"# result is {selected_ids}")
        selected_ids = json.dumps(selected_ids) #+ "\n"
        print(f"# sending '{selected_ids}'")
        self.wfile.write(bytes(selected_ids, "utf-8"))
        self.wfile.flush()

##Open port on HOST   
def open_port():
    with socketserver.TCPServer((HOST, PORT), ExampleHandler) as server:
        print(f"# opening port '{PORT}'")
        server.serve_forever()

#if __name__ == "__main__":
    ##First parse arguments and set up global variables
    #python simulator.py -pc 44 -d rmab_cog_sim -l 60 -s 0 -ws 0 -ls 0 -g 0.99 -S 2 -A 2 -n 10 -lr 1 -N 1 -sts -1 -b 0.2
    
##Finally, open port for communication
open_port()

    ### DREW: DON'T WORRY ABOUT SAVING ANYTHING RIGHT NOW...COMMENTED OUT #########
'''
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
        
        
        commented out by Shahin
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
