RMAB edited for running CogModel Simulation
Many options have been removed from the main simulator.py file as well as any code to save data and plot results

You need to go to the src folder and run simulator.py. If you are using python3 then it would be something like this. This file has a lot of inputs that you need to tune:

python3 simulator.py -pc 44 -d rmab_cog_sim -l 60 -s 0 -ws 0 -ls 0 -g 0.99 -S 2 -A 2 -n 10 -lr 1 -N 1 -sts -1 -b 0.2

-pc: policy option (set to 44) 

-d: dataset (set to rmab_cog_sim)

-l: number of rounds (set to 60)

-S: number of states (set to 2 currently, or 3 depening on whether we have the intermediate state or not)

-A: number of actions (set to 2)

-n: number of participants (set to 10) if you change this number, you need to edit it both in simulator and also simulation_environment

-N: number of repeats of each trial (set to 1) you will get a run-time warning when setting N to 1

-g: discount factor (set to 0.99 -- default is 0.95)

-sts: start state (set to -1 for random)

-b: budget (set to 0.2, as in 20% budget)

-lr: learning option (set to 1 for thompson sampling)

-s: random seed starting state (set to 0)

-ws: random world seed starting state (set to 0)

-ls: random learning seed starting state (set to 0)
