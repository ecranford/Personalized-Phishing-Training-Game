This is an example Cog Model that needs to be modified to communicate directly with the RMAB to run simulations.
combinedemailsdata_cleaned.csv is the database of emails, but needs to be updated based on edits made to the PEST emails (and to some of the PTT emails).

#|
MURI Phishing Model -- MAB study

Drew Cranford
Don Morrison
Christian Lebiere

;; The model defines 1000 models to run 1000 agents simultaneously. The main entry
;; point here is (run-task). It first loads files to write data to, one for the
;; initialization data, and another for the main task results. It then loads the emails
;; into an a-list. It then initializes the model with a random number of emails, uniformly
;; distributed, and random ratio of ham to phishing emails, normally distributed. Then it
;; runs the agents through n trials. On each trial, the (run-trial) first calls
;; (select-users) to determine what users to send phishing emails to. The default is 15%
;; of users selected randomly from a uniform distribution. The model then runs each agent
;; one at a time, generates a decision, saves the data to file (and to an a-list that 
;; tracks user-history), provides feedback to the agent if it incorrectly classified
;; a phishing email, and runs the agent forward 30s. The process repeats for n trials.

10/25/2018 -- creation date
04/08/2019 -- modified model for ICCM paper: parameters same as conceptual model
07/24/2019 -- adapted model to run against Kuldeep's dataset 
              60 trials total (10 pre-train, 40 train w/feedback, and 10 test trials)
08/01/2019 -- created v2 of Kuldeep model; see below for details
04/14/2022 -- modified Kuldeep model to run MAB study; combines PTT and PEST databases

v1.0 = randomized initialization (length & ratio), bll 0.5, ans 0.25, temp nil, mp 2.0
v1.3 = v1.0 + higher ham ratio initialization
v1.4 = v1.3 + even higher ham ratio initialization
v1.5 = v1.4 + new design with 20 pre-test, 60 training, 20 post-test

random = random selection of users to receive email (baseline 15% selection rate)
random20 = random 20% of users
test = test with phishing email between each trial (for 2 states)
test2 = test with two phishing emails between each trial (for 3 states)
|#
