ADD RMAB files to this directory and then update this README

You need to go to the src folder and run simulator_seq.py. If you are using python3 then it would be something like this. This file has a lot of inputs that you need to tune:

python3 simulator_seq.py -pc -74 -d simple_spam_ham -l 100 -s 0 -ws 0 -ls 0 -g 0.99 -S 3 -A 2 -n 1000 -lr 1 -N 2 -sts -1 -b 0.2

-pc: policy option use 74 
-d: dataset
-l: number of rounds
-S: number of states
-A: number of actions
-n: number of participants
