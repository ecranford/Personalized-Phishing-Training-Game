ADD RMAB files to this directory and then update this README

You need to go to the src folder and run simulator_seq.py. If you are using python3 then it would be something like this. This file has a lot of inputs that you need to tune:

python3 simulator_seq.py -pc -74 -d behavorial -l 100 -s 0 -ws 0 -ls 0 -g 0.99 -S 3 -A 2 -n 4 -lr 1 -N 1 -sts -1 -b 0.2

-pc: policy option (set to 74) 

-d: dataset (set to behavorial)

-l: number of rounds (set to 60)

-S: number of states (set to 2 or 3 depening on whether we have the intermediate state or not)

-A: number of actions (set to 2)

-n: number of participants (set to 4) if you change this number, you need to edit it both in simulator_seq and also simulation_environment

-N: number of repeats of each trial (set to 1) you will get a run-time warning when setting N to 1

-g: discount factor (set to 0.99 -- default is 0.95)

-sts: start state (set to -1 for random)
