Cog Model has been modified to communicate directly with the RMAB to run simulations.
MURI_PhishingModel_RMAB_Simulations.lisp is the cog model simulation
combinedemailsdata_cleaned_updated_20221122.csv is the database of emails
similarity-cache.lisp is the cache of similarities between features of emails
EmailSimilarityMatrix_sum_decomp.csv is the cache of summed similarities for model-tracing verstion of the cog model
similarity-service.lisp is the service that gets similarities (will be cached for this simulation).

Data will be saved as:
MURI_PhishingModel_MAB_v1.7_RMAB_Sims_InitData.txt is the intialization specs for all participants
MURI_PhishingModel_MAB_v1.7_RMAB_Sims.txt is the trial-to-trial data for all participants

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
v1.4_CogSelect2 = 1.4 + Cognitive Selection Algorithm v2
v1.4_CogSelect3 = v1.4 + Cog Selection Algorithm v3 (similarities based on local avgs)
v1.4_CogSelect4 = v1.4_CS3 + only based on probabilities given phishing email
v1.5_CogSelect4 = v1.4_CS4 + new design with 20 pre-test, 60 training, 20 post-test
v1.7_CogSelect4 = v1.5_CS4 + new initialization, updated similarities, & new procedure
  
random = random selection of users to receive email (baseline 15% selection rate)
random20 = random 20% of users
|#  

