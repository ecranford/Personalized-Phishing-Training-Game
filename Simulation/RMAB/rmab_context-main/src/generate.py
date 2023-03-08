import math
import pandas as pd
import numpy as np
import mdptoolbox as mtb
import matplotlib.pyplot as plt
# import simulation_environments as se

def rmab_context_features(N, K=1):
    F = np.array(sorted(np.random.random_sample(N)))
    # F = np.array([0.3, 0.301, 0.305, 0.309])
    T = []
    R = []
    TMOD = []
    RMOD = []

    for i in range(N):
        fi = F[i]
        # ben1 = np.random.uniform(0.01, 0.1)
        # ben2 = np.random.uniform(0.01, 0.1)
        ben1 = 0.1
        ben2 = 0.1

        p000 = fi + 0.5
        # p000 = fi + 0.15
        if p000 > 0.9:
            p000 = 0.9
        p010 = p000 - ben1

        p100 = fi + 0.2
        # p100 = fi + 0.11
        if p100 > 0.7:
            p100 = 0.7

        p110 = p100 - ben2

        t = np.array([[[p000, 1-p000],  # for state B action 0
                       [p010, 1-p010]],  # for state B action 1

                      [[p100, 1-p100],  # for state G action 0
                        [p110, 1-p110]]  # for state G action 1
                      ])
        
        tmod = np.array([[[p000, 1-p000], # a = 0
                          [p100, 1-p100]], # a = 0
                          
                         [[p010, 1-p010], # a = 1
                          [p110, 1-p110]] # a = 1
                        ])

        rmod = np.array([[[0, 1],
                          [0, 1]],
                          
                         [[0, 1],
                          [0, 1]]
                        ])
        # print(t)
        r = [0, 1]

        T.append(t)
        R.append(r)
        TMOD.append(tmod)
        RMOD.append(rmod)

    T = np.array(T)
    TMOD = np.array(TMOD)
    R = np.array(R)
    RMOD = np.array(RMOD)
    C = np.array([0, 1])
    F = F / K

    return T, R, C, F, TMOD, RMOD


ntrials = 1
narms = 5000
epochs = 1e4
nstates = 2
nactions = 2

epsilon = 0.1
gamma = 0.99

seedbase = 0

np.random.seed(seed=seedbase)
world_random_stream = np.random.RandomState()
world_random_stream.seed(seedbase)
T, R, C, F, Tmod, Rmod = rmab_context_features(narms)

# np.save("F2.npy", F)

qtable = np.zeros((narms, 2, 2))

vals = [[[], []],
        [[], []]]

diff = [[[], []],
        [[], []]]

numer = [[[], []],
        [[], []]]

LC = np.zeros((2, 2))


# for i in range(narms):

#     # state = np.random.randint(0, nstates)
#     state = 0

#     counts = np.zeros((nstates, nactions))

#     # print(Rmod[i].shape)
#     ql = mtb.mdp.QLearning(Tmod[i], Rmod[i], gamma, epochs)
#     ql.run()
#     # print(ql.Q)
#     qtable[i] = ql.Q

#     if i % 100 == 0:
#         np.save("Q2.npy", qtable)

qtable = np.load("../logs/Q/0/Q.npy")
# print(qtable.nonzero())
# samples = 1400
# qtable = qtable[:samples]
# F = F[:samples]
F = np.load("../logs/F/0/F.npy")

for i in range(len(qtable)):
# for i in range(narms):
    for j in range(i):
        if j == i:
            break
        if abs(F[i] - F[j]) < 1e-1:
            continue
        for s in range(2):
            a = 1
            temp = abs((qtable[i][s][a] - qtable[j][s][a]/(F[i]-F[j])))

            vals[s][a].append(temp)
            diff[s][a].append(abs(F[i] - F[j]))
            numer[s][a].append(abs(qtable[i][s][a] - qtable[j][s][a]))

            LC[s,a] = max(LC[s,a], temp)

import pdb
pdb.set_trace()
print(vals)

for s in range(2):
    # for a in range(2):
    a = 1
    print(s, a, max(vals[s][a]))

# for s in range(2):
#     for a in range(2):
#         plt.figure()
#         print(F.shape)
#         print(qtable[:,s,a].shape)
#         m, b = np.polyfit(F, qtable[:,s,a], 1)
#         plt.scatter(F, qtable[:,s,a], s=1)
#         plt.plot(F, m*F + b)
#         plt.xlabel("Features")
#         plt.ylabel("Q("+str(s) + "," + str(a) + ")")
#         plt.savefig("seed0_Q_"+str(s) + str(a))
    
# plt.figure()
# plt.scatter(numer[s][a], diff[s][a], s=1)
# plt.scatter(diff[s][a], numer[s][a], s=1)
# xt, xtl = plt.xticks()
# xt=np.append(xt, 1e4)

# xtl=np.append(xtl, "1e4")
# plt.xticks(xt, xtl)
# plt.ylim((0, 1e-5))
# plt.ylabel("Ranges of Q(i) - Q(j)")
# plt.xlabel("D(i,j)")
# plt.savefig("QDiffvsDiff_seed0_s1.png")

# for s in range(2):
#     for a in range(2):
#         print(sorted(vals[s][a]))
# import pdb
# pdb.set_trace()
# np.save("vals.npy", vals)
# np.save("LC.npy", LC)
# print(len(vals[0][1]))
# print(np.mean(np.array(vals[0][1])))
# print(np.percentile(np.array(vals[0][1]), 95))
# print(np.percentile(np.array(vals[0][1]), 97))
# print(np.percentile(np.array(vals[0][1]), 100))
# bins=[0, 5, 10, 50, 100, 500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8]
# tl=["0", "5", "10", "50", "100", "500", "1000", "5000", "1e4", "5e4", "1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8", "5e8"]
# # labels = ["5-10", "10-50", "50-100", "100-500", "500-1000", "1000-5000", "5000-10000", 
# #         "10000-50000", "50000-100000", "100000-500000", "500000-1000000", "1000000-5000000"]
# labels = [str(tl[i]) + "-" + str(tl[i + 1]) for i in range(len(bins) - 1)]
# bins=[0, 5, 10, 50, 100, 500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8]
# df = pd.DataFrame(diff[0][1],columns=["a"])
# ax = df.groupby(pd.cut(df.a, bins=[0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4])).size().plot.bar(rot=45)
# ax.set_xlabel("Ranges of D(i,j)")
# ax.set_ylabel("Count")
# plt.savefig("DiffPlot.png", bbox_inches='tight')
# ax = df.groupby(pd.cut(df.a, bins=bins, labels=labels)).size().plot.bar(rot=45)

# print(vals[0][1])
# fig = plt.figure()
# plt.bar(bins, temp2)
# plt.hist(vals[0][1], bins="auto", labels=bins)

# plt.xticks([0, 100, 200, 500, 1000, 5000, 10000])
# plt.ioff()

# mu = np.array(vals[0][1]).mean()
# maxval = np.array(vals[0][1]).max()
# val97 = np.array(np.percentile(np.array(vals[0][1]), 97))


# textstr = 'mean=' + "{:.2e}".format(mu) +"\n" +\
#             'max=' + "{:.2e}".format(maxval) +"\n"+\
#             '97%=' + "{:.2e}".format(val97)

# # these are matplotlib.patch.Patch properties
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# # place a text box in upper left in axes coords
# ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
#         verticalalignment='top', bbox=props)

# ax.set_xlabel("Ranges of l(i,j)")
# ax.set_ylabel("Count")

# plt.savefig("NewVals01.png", bbox_inches='tight')
# print(max(F), min(F))

# plt.cla()
# plt.close()

# plt.plot()
# plt.hist(vals[1][1], bins='auto')

# plt.savefig("Vals11.png")

    # for j in range(epochs):
    #     if np.random.uniform() < epsilon:
    #         action = np.random.randint(0, nactions)
    #     else:
    #         action = np.argmax(qtable[i][state])

    #     # counts[state][action] += 1
    #     # alpha = 1/(counts[state][action] + 1)
    #     # alpha = 0.1
    #     alpha = 1/math.sqrt(j + 2)
        
    #     # next_state = np.argmax(world_random_stream.multinomial(1, T[i, state, int(action), :]))
    #     p_s_new = np.random.random()
    #     p = 0
    #     s_new = -1
    #     while (p < p_s_new) and (s_new < (nstates - 1)):
    #         s_new = s_new + 1
    #         p = p + T[i, state, action, s_new]

    #     next_state = s_new

    #     reward = R[i][next_state]
    #     # print(state, action, next_state, reward + gamma * max(qtable[i][next_state]))
    #     qtable[i][state][action] += alpha*(reward + gamma * max(qtable[i][next_state]) - qtable[i][state][action])

    #     # print(qtable[i])
    #     state = next_state
    #     # sleep(0.8)

    # print(qtable[i], F[i])

    # print(T)
    # for j in range(i):
    #     if j == i:
    #         break
    #     for s in range(2):
    #         # for a in range(2):
    #             a = 1
    #             temp = abs((qtable[i][s][a] - qtable[j][s][a]/(F[i]-F[j])))
    #             vals[s][a].append(temp)

    #             if temp > LC[s, a] or i == 1:
    #                 print("######")
    #                 print(i, j)
    #                 print(temp, F[i], F[j], F[i] - F[j])
    #                 print(T[i])
    #                 print(T[j])
    #                 print(s, a)
    #                 print(qtable[i])
    #                 print(qtable[j])
    #                 print("#####")


    #             LC[s,a] = max(LC[s,a], temp)
    #             print(T[i], T[j])
    
    # print(LC)
    # if i == 1:
    #     break