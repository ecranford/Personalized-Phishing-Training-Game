;;; Copyright (c) 2018 Carnegie Mellon University
;;;
;;; Permission is hereby granted, free of charge, to any person obtaining a copy of this
;;; software and associated documentation files (the "Software"), to deal in the Software
;;; without restriction, including without limitation the rights to use, copy, modify,
;;; merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
;;; permit persons to whom the Software is furnished to do so, subject to the following
;;; conditions:
;;;
;;; The above copyright notice and this permission notice shall be included in all copies
;;; or substantial portions of the Software.
;;;
;;; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
;;; INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
;;; PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
;;; HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
;;; CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
;;; OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#|
Personalized Phishing Training selection algorithm -- Cog EV

Drew Cranford
Don Morrison
Christian Lebiere

;;; The model selects users to send phishing emails to by weighing the anticipated future
;;; benefits of sending a phishing email, in terms of correctly classifying phishing emails, 
;;; against the anticipated future costs, in terms of incorrectly classifying ham emails, 
;;; to determine which users will most benefit from a phishing training intervention. The
;;; input to the model is a history of users' classification decisions that include the
;;; user-id, email-id, classification decision, and time of decision. The model gets the
;;; email features from a database based on the email-id. Similarities between features
;;; are computed using the UMBC semantic similarity tool, but which have been precomputed
;;; and saved to a hash table for easy lookup. Output is a list of user IDs.

9/26/2023 -- creation date

v1.0 = Cog Low baseline method

|#

(in-package :common-lisp-user)

;;Load similarity service
;;;;DON'T THINK I NEED THIS IF SIMILARITIES ARE PRECOMPUTED AND SAVED IN *EmailSims*
;;(load "similarity-service.lisp")
;;(ss:load-semantic-textual-similarity-cache)

;;Load packages
;;; NOTE: need to remove packages that aren't used and add packages necessary for
;;; communication with experiment
(ql:quickload '(:alexandria
		:cl-utilities
		:cl-ppcre
		:external-program
		:vom
		:unix-options
		:cl-csv
		:cl-json
		:split-sequence
		:usocket-server))

(import '(alexandria:if-let alexandria:when-let cl-utilities:n-most-extreme))

(vom:config t :debug)

;;(defconstant +default-host+ "localhost" :test #'equal)
(defconstant +default-port+ 9142)

(defparameter +seed+ '(987654321 3)) ;;seed for creating executable

(defparameter *EmailData* '())
(setf *EmailData* '())
(defparameter *EmailSims* '())
(defparameter *user-history* '())
(setf *user-history* '())
(defparameter *user-list* '())
(setf *user-list* '())
(defparameter *user-data* '())
(setf *user-data* '())
;;(defparameter *pHam* 0.0)
;;(defparameter *pPhishing* 0.0)
(defparameter *email-list* '())
(setf *email-list* '())

;;;ADD this somewhere in beginning of start function to read in email similarities
;;Read in Email Similarity data into a-list
(cl-csv:do-csv (row (make-pathname :type "csv" :name "EmailSimilarityMatrix_sum") :skip-first-p t)
  (let ((email-id (read-from-string (first row))))
    (setf *EmailSims* (acons (read-from-string (first row))
			     nil
			     *EmailSims*))
    (dotimes (i 367)
      (setf (cdr (assoc email-id *EmailSims*)) (acons (1+ i)
						      (read-from-string (nth (+ 1 i) row))
						      (cdr (assoc email-id *EmailSims*)))))))

;;create file to save data to if does not already exist
;;Open file to write initi data to
(with-open-file (str (make-pathname :type "txt" :name "data/coglow_data")
		     :direction :output
		     :if-exists nil
		     :if-does-not-exist :create)
  (format str "USER~CTrial~CPIphish~CPIham~CPIdiff~%" #\Tab #\Tab #\Tab #\Tab))

(defun approx-act-r-noise (s)
  "Approximates a sample from a normal distribution with mean zero and
   the given s-value (/ (sqrt (* 3.0 variance)) 3.1416)."
  ;; Need to test bound because of short-float lack of precision
  (if (and (numberp s) (plusp s))
      (let ((p (max 0.0001 (min (random 1.0) 0.9999))))
        (* s (log (/ (- 1.0 p) p))))
      (format t "Act-r-noise called with an invalid s ~S" s)))

;;select-users will return a list of users to send phishing emails to (default: select randomly)
;;CogSelect will select 20% of users

(defparameter +mull+ 0)          ;;mulling
(defparameter +blc+ 5.0)         ;;base-level activation constant
(defparameter +bll+ 0.5)         ;;base-level learning (decay parameter)
(defparameter +mp+ 2.0)          ;;mismatch penalty parameter for partial matching
(defparameter +ans+ 0.25)        ;;transient noise parameter
(defparameter +tmp+ nil)         ;;temperature parameter
(if (not +tmp+) (setf +tmp+ (* (sqrt 2) +ans+)))

#|
activation equation = bla + pm + noise
bla = blc + ln(sum(t^-bll))
partial matching = sum(mp*(sim(k,i)))
noise = (act-r-noise ans)
therefore a = blc + ln(sum(t^-bll)) + sum-of-all-slots-k(mp*(sim(k,i))) + (act-r-noise ans)

Pi = exp(Ai/(* (sqrt 2) ans)) / sum-of-all-chunks-j(exp(Aj/(* (sqrt 2) ans)))
|#

(defun select-users (current-time user-history)
  (let ((user-ids '())
	(user-data '()) ;;a-list to store ((user-id . (piphish . piphish) (piham . piham)))
	(trial (- (1+ (length (cdar (first user-history)))) 20)))
    (dolist (user (copy-seq user-history))
      (let ((user-id (caar user))
	    (instance-list (cdar user))) ;;need to change all incorrect phishing classifications to classification=phishing
	(vom:debug "user-id = ~S" user-id)
	;;(vom:debug "instance-list = ~S" instance-list)
	(cond (instance-list
	       ;;determine current probability to classify a phishing email correctly
	       (vom:debug "Compute prob to classify phishing email correctly")
	       (let ((instances '())) ;;a-list to store ((email-id . (classification . class) (activation . ai) (prob-retrieval . pi)))
		 (dolist (instance (copy-seq instance-list))
		   (let* ((phase (cdr (assoc 'phase instance)))                               ;;phase (string)
			  (email-id (parse-integer (cdr (assoc 'email--id instance))))        ;;email-id (integer)
			  (email-type (cdr (assoc 'email--type instance)))                    ;;email-type (string)
			  (feedback (if (and (equal "phase 2" phase)                          ;;if feedback given (boolean)
					     (equal "PHISHING" email-type)
					     (equal "HAM" (cdr (assoc 'classification instance))))
					t
					nil))
			  (class (if feedback                                                 ;;classification (symbol)
				     'phishing
				     (read-from-string (cdr (assoc 'classification instance)))))
			  (stime (cdr (assoc 'timestamp instance)))                           ;;time of storage (integer)
			  (tj (/ (- (+ current-time 15000) stime) 1000)) ;;time (in sec) since last presentation
			  )		     
		     ;;(vom:debug "email-id = ~S (~S) (~S)" email-id class stime)
		     ;;compute activation of each email using partial matching
		     (setf instances (acons email-id
					    (pairlis '(classification activation)
						     `(,class
						       ,(+ +blc+
							   (log (if feedback              ;;if feedback t, then add mulling n times
								    (+ (* +mull+ (expt (- tj 5) (- +bll+)))
								       (expt tj (- +bll+)))
								    (expt tj (- +bll+))))
							   (* +mp+
							      (cdr (assoc 366 (cdr (assoc email-id *EmailSims*)))))
							   ;;(approx-act-r-noise +ans+)
							   )))
					    instances)) ;;compute activation
		     ))
		 ;;compute retrieval probability of each email (boltzmann equation)
		 ;;first collect activations into a-list and sum them
		 (let ((boltzmann-denom (loop for instance in (copy-seq instances)
					      sum (exp (/ (cdr (assoc 'activation (cdr instance))) +tmp+)))))
		   ;;then compute boltzmann equation on each instance and add to instances list
		   (dolist (inst (copy-seq instances))
		     (setf (cdr (assoc (car inst) instances)) (acons 'prob-retrieval
								     (/ (exp (/ (cdr (assoc 'activation (cdr inst))) +tmp+))
									boltzmann-denom) ;;compute prob-retrieval
								     (cdr (assoc (car inst) instances))))
		     ))
		 ;;compute pi phishing and pi ham and add to user-data
		 (setf user-data (acons user-id
					(pairlis '(pi-phish pi-ham)
						 `(,(loop for instance in (copy-seq instances)
							  when (equal (cdr (assoc 'classification (cdr instance))) 'phishing) ;;pi phishing
							    sum (cdr (assoc 'prob-retrieval (cdr instance))))
						   ,(loop for instance in (copy-seq instances)
							  when (equal (cdr (assoc 'classification (cdr instance))) 'ham) ;;pi ham
							    sum (cdr (assoc 'prob-retrieval (cdr instance))))))
					user-data)))
	       
	       
	       ;;select users to maximize probability of correct phishing classifications
	       ;;pi-phish - pi-ham
	       (vom:debug "Compute PIphish")
	       ;;(vom:debug "User ~S data is ~S" user-id user-data)
	       (let ((pi-diff (- (cdr (assoc 'pi-phish (cdr (assoc user-id user-data))))
				  (cdr (assoc 'pi-ham (cdr (assoc user-id user-data)))))))
		 (setf (cdr (assoc user-id user-data)) (acons 'PI-DIFF
							      pi-diff
							      (cdr (assoc user-id user-data)))))
	       )
	      (t
	       (setf user-data (acons user-id
				      `(,(cons 'PI-DIFF 0.5))
				      user-data))))
	))

    ;;save data to file
    (with-open-file (str (make-pathname :type "txt" :name "data/coglow_data")
			 :direction :output
			 :if-exists :append
			 :if-does-not-exist :create)
      (dolist (user user-data) 
	(format str "~A~C~A~C~A~C~A~C~A~%"
		(car user) #\Tab trial #\Tab (cdr (assoc 'pi-phish (cdr user))) #\Tab (cdr (assoc 'pi-ham (cdr user))) #\Tab (cdr (assoc 'pi-diff (cdr user))))))
    
    ;;(vom:debug "user-data is ~S" user-data)
    ;;select set of users with highest EVdiff values
    (setf user-ids (mapcar #'car (n-most-extreme 2 user-data #'< :key #'(lambda (x) (cdr (assoc 'PI-DIFF (cdr x)))))))
    
    user-ids)
  )


;;;COMMUNICATION FUNCTIONS HERE
;;input will be json list of users and their data
;;need to construct *user-list* and *user-history* from this json
;;what is *user-data* for? do I need it?
;;return array of user-ids

(defun process-request (stream)
  (handler-case
      (progn
	(vom:debug "Connection opened")
	(with-input-from-string (msg (read-line stream))
	  (vom:debug "Recieved message ~S" msg)
	  (let ((json:*json-symbols-package* nil))
	    (let* ((json-msg (json:decode-json msg))
		   (ctime (cdr (assoc 'current--time json-msg)))
		   (user-history (cdr (assoc 'player--data json-msg))))
	      ;;(vom:debug "json message is ~S" json-msg)
	      ;;(vom:debug "currrent-time = ~S" ctime)
	      ;;(vom:debug "user-history = ~S" user-history)
	      (let ((user-list (mapcar (lambda (s) (subseq (symbol-name s) 2)) (select-users ctime user-history)))) ;;select-user
		(vom:debug "Sending ~S" user-list)
		(format stream "[~{~S~^,~}]~%" user-list)
		(finish-output stream))))))
    (error (e) (vom:error "Error processing request: ~A" e)))
  )

;;If list is a list of strings, then(mapcar (lambda (s) (subseq(s, 2)) list)will be a list of the relevant integers.
;;(mapcar (lamba (s) (subseq s 2)) list)

(usocket:socket-server nil +default-port+ #'process-request)

;;Functions to save and executable
(defun mainp ()
  (with-simple-restart (abort "Aborted")
    (unix-options:with-cli-options () (reproducible)
      (let ((seed (cond (reproducible +seed+)
                        (t (setf *random-state* (make-random-state t))
                           (list (random most-positive-fixnum) (random 100))))))
        (vom:info "Seed = ~A" seed)))
    (select-users)))

#+sbcl
(defun save-exec (&optional (filename "MURI-PPT-standalone"))
  (sb-ext:save-lisp-and-die filename :executable t :toplevel #'mainp))

#+ccl
(defun save-exec (&optional (filename "MURI-PPT-standalone-actr-model"))
  (ccl:save-application filename :prepend-kernel t :toplevel-function #'mainp))
