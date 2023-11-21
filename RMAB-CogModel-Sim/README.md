# Personalized-Phishing-Training-Game
CogModel Simulation for Personalized Phishing Training Game

*Simulation* directory contains files for running the simulation experiment using the Cog Model as simulated humans and communication between the RMAB (python) and Cog Model (Lisp) done over TCP using JSON.

To start the simulation:
1) open a terminal to the './RMAB' directory and run the following command (change 'python' to whatever the python command is on your system...e.g., usually 'python3'):
   python simulator.py -pc 44 -d rmab_cog_sim -l 60 -s 0 -ws 0 -ls 0 -g 0.99 -S 2 -A 2 -n 10 -lr 1 -N 1 -sts -1 -b 0.2

2) open a terminal to the './CogModel' directory and run the following commands, assuming that the lisp implementation sbcl is installed and actr has been downloaded (the location of the load-act-r.lisp file may be different on your machine):
   sbcl --dynamic-space-size 13312 --load "C:\actr7\load-act-r.lisp"

3) After ACT-R loads up, load up the model by typing the following and then press enter:
   (load "MURI_PhishingModel_RMAB_Simulations.lisp")

4) After the model loads up, type the following to run the model for t trials for n number of groups with 10 participants per group (e.g., the following code will run the model for 100 trials...20 pretest, 60 training, and 20 posttest...for 100 groups of 10 participants...1000 participants total):
   (run-task 100 100)

**Recommend running only 1 group if just testing things, and 100 for analyzing data.
