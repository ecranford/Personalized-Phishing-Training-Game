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

(in-package :common-lisp-user)


;;Blending Module Functions
(require-extra "blending") ;; gets the blending module if not already loaded

;; redefine function to grab values from blending module -- thanks Dan
(defun cumulative-values ()
  (let ((b (get-module blending))
        (results nil))
    (when b
      (maphash (lambda (time sblt)
                 (let ((current (list time)))
                   (unless (sblt-trace-no-matches sblt)
                     (let ((slots-data nil))
                       (dolist (slot (sblt-trace-slot-list sblt))
                         (let ((slot-data (list (sblt-slot-name slot))))
                           (case (sblt-slot-condition slot)
                             (:null)
                             (:numbers
                              (let ((sum 0))
                                (dolist (mag (sblt-slot-magnitudes slot))
                                  (awhen (blending-item-mag mag)
				    (let ((increment (* it
							(blending-item-prob mag))))
				      (incf sum increment))))
				(if (sblt-slot-mag-adjusted slot)
                                    (push-last (list (sblt-slot-adjusted-val slot) nil) slot-data)
				    (push-last (list sum nil) slot-data))))
                             ((:chunks :other :not-generalized)
                              (if (= 1 (length (sblt-slot-chunks slot)))
                                  (push-last (list (car (sblt-slot-chunks slot)) nil) slot-data)
				  (dolist (val (sblt-slot-chunks slot))
				    (let ((sum 0.0))
				      (dolist (possible-list (mapcar 'cdr (remove-if-not
									   (lambda (x) (eq (car x) val))
									   (sblt-slot-possible-values slot))))
					(let ((possible (first possible-list))
					      (sim (second possible-list)))
					  (incf sum (* (blending-item-prob possible) (expt sim 2)))))
				      (push-last (list val sum) slot-data))))))
                           (push-last slot-data slots-data)))
                       (push-last slots-data current)))
                   (push current results)))
               (blending-module-trace-table b))
      results)))

(defun confidence-values ()
  (let ((b (get-module blending))
        (results nil))
    (when b
      (maphash (lambda (time sblt)
                 (let ((current (list time)))
                   (unless (sblt-trace-no-matches sblt)
                     (dolist (slot (sblt-trace-slot-list sblt))
                       (case (sblt-slot-condition slot)
                         (:null)
                         (:numbers)
                         ((:chunks :other :not-generalized)
                          (if (= 1 (length (sblt-slot-chunks slot)))
                              (push-last (list (sblt-slot-name slot) (car (sblt-slot-chunks slot)) 1.0) current)
                            (let ((min-val nil)
                                  (min-result nil))
                              (dolist (val (sblt-slot-chunks slot))
                                (let ((sum 0.0))
                                  (dolist (possible-list (mapcar 'cdr (remove-if-not (lambda (x) (eq (car x) val))
										     (sblt-slot-possible-values slot))))
                                    (let ((possible (first possible-list))
                                          (sim (second possible-list)))
                                      (incf sum (* (blending-item-prob possible) (expt sim 2)))))
                                  (when (or (null min-val) (< sum min-val))
                                    (setf min-val sum)
                                    (setf min-result val))))
                              (push-last (list (sblt-slot-name slot) min-result (- 1.0 min-val)) current))))))
                     (push current results))))
               (blending-module-trace-table b))
      (sort results #'< :key 'car))))

;;;;Run in terminal with act-r loaded (alternatively run in folder and uncomment next two lines)
;;#-act-r
;;(load "~/actr7/load-act-r.lisp")

;;Load similarity service
(load "similarity-service.lisp")
(ss:load-semantic-textual-similarity-cache)

;;Load packages
(ql:quickload '(:alexandria
		:cl-utilities
		:cl-ppcre
		:external-program
		:vom
		:unix-options
		:cl-csv
		:split-sequence
		:usocket
		:cl-json))

(import '(alexandria:if-let alexandria:when-let alexandria:shuffle cl-utilities:n-most-extreme))

(defparameter +host+ "127.0.0.1")
(defconstant +default-port+ 9143)

;;use seed to reproduce results
(defparameter +seed+ '(987654321 3))

(defparameter *request-index* 0)

;;(defvar *similarity-engine* nil)
(defparameter *EmailData* '())
(setf *EmailData* '())
(defparameter *EmailSims* '())
(setf *EmailSims* '())
(defparameter *init-list* '())
(setf *init-list* '())
(defparameter *user-history* '())
(setf *user-history* '())
(defparameter *obs-states* '())
(setf *obs-states* '())
(defparameter *user-list* '())
(setf *user-list* '())
(defparameter *user-data* '())
(setf *user-data* '())
(defparameter *group-id* 0)
(setf *group-id* 0)
;;(defparameter *pHam* 0.0)
;;(defparameter *pPhishing* 0.0)
(defparameter *email-list* '())
(setf *email-list* '())
(defparameter *pre-test-list* '())
(setf *pre-test-list* '())
(defparameter *post-test-list* '())
(setf *post-test-list* '())
(defparameter *mull-lim* 60)
(defparameter *mull-list* '())

;;functions to decompose url's and email addresses if desired (not used by default)
(defun decompose-sender (s)
  (nsubstitute-if #\Space #'(lambda (c) (member c '(#\_ #\- #\@ #\.))) s)
  s)
(defun decompose-link (s)
  (nsubstitute-if #\Space #'(lambda (c) (member c '(#\% #\? #\= #\_ #\& #\- #\/ #\: #\@ #\.))) s)
  s)

(defun simplify-sender (s)
  (let ((result (cl-ppcre:regex-replace-all " \\b(com)\\b"
					    (cl-ppcre:regex-replace-all " \\b(edu)\\b"
									(cl-ppcre:regex-replace-all " \\b(org)\\b"
												    (decompose-sender s) "") "") "")))
    result))

(defun simplify-link (s)
  (let ((result (if (ppcre:scan-to-strings "www.*" s)
		    (subseq (ppcre:scan-to-strings "www..*" s) 4)
		    (if (ppcre:scan-to-strings "://.*" s)
			(subseq (ppcre:scan-to-strings "://.*" s) 3)
			s))))
    (decompose-link result)))

(defun approx-act-r-noise (s)
  "Approximates a sample from a normal distribution with mean zero and
   the given s-value (/ (sqrt (* 3.0 variance)) 3.1416)."
  ;; Need to test bound because of short-float lack of precision
  (if (and (numberp s) (plusp s))
      (let ((p (max 0.0001 (min (random 1.0) 0.9999))))
        (* s (log (/ (- 1.0 p) p))))
      (format t "Act-r-noise called with an invalid s ~S" s)))

;;function to save init data
(defun save-init-data (user num-init ham-ratio num-ham-init dm-count email-id class)
  (with-open-file (str (make-pathname :type "txt" :name "MURI_PhishingModel_MAB_v1.7_RMAB_Sims_InitData")
			       :direction :output
			       :if-exists :append
			       :if-does-not-exist :create)
	    (format str "~A~C~A~C~A~C~A~C~A~C~A~C~A~%"
		    user #\Tab num-init #\Tab ham-ratio #\Tab num-ham-init #\Tab dm-count #\Tab email-id #\Tab class))
  )

;;select-users will return a list of users to send phishing emails to (default: 20% budget)
(defun select-users (trial group-id obs-states)
  (let ((user-ids '()))
    (cond ((<= 20 trial 79)
	   (let* ((socket (usocket:socket-connect +host+ +default-port+))
		  (request (pairlis '(trial group-id obs-states) `(,(- trial 20) ,group-id ,obs-states))))
	     (unwind-protect
		  (let ((stream (usocket:socket-stream socket)))
		    (format t ";; request: ~S~%" request)
		    (setf request (json:encode-json-alist-to-string request))
		    (format t ";; sending: ~S~%" request)
		    (format stream "~A~%" request)
		    (finish-output stream)
		    (setf user-ids (read-line stream))
		    (format t ";; received: ~A~%" user-ids)
		    (setf user-ids (mapcar (lambda (x) (read-from-string x))
					   (json:decode-json-from-string user-ids)))
		    (format t ";; decoded response: ~S~%" user-ids)
		    )
	       (usocket:socket-close socket))))
	  ((< trial 20) ;;select users based on pre-test list
	   (dolist (user-data *pre-test-list*)
	     (if (equal (nth trial (cdr user-data)) 'phish)
		 (push (car user-data) user-ids))))
	  (t ;;select users based on post-test list
	   (dolist (user-data *post-test-list*)
	     (if (equal (nth (- trial 80) (cdr user-data)) 'phish)
		 (push (car user-data) user-ids))))
	  )
    user-ids)
  )

;;run-trial will run the models through 1 trial of the task
(defun run-trial (trial group-id user-list)
  ;;determine who to send phishing emails to
  (let ((selected-users (select-users trial group-id (copy-seq *obs-states*)))
	(email-list '())
	(phase (cond ((<= 20 trial 79)
		      'training)
		     ((< trial 20)
		      'pre-test)
		     (t 'post-test))))
    
    ;;simulate reading email (set goal to class-email)
    (dolist (user user-list)
      (with-model-eval user
	(goal-focus-fct 'read-email)))
    (run-full-time 15.0)
    
    ;;run models
    (dolist (user user-list)
      ;;(format t "user ~A email-ids: ~A~%" user (mapcar #'car (cdr (assoc user *user-history*))))
      (let* ((email-ids (mapcar #'car (cdr (assoc user *user-history*))))
	     (email-id (let ((id nil))
			 (if (member user selected-users) ;;if user should get a phishing email select phishing, else ham
			     (loop while (let ((p-id (+ 1 (random 188))))
					   (setf id p-id)
					   (member p-id email-ids)))
			     (if (>= (count-if #'(lambda (x) (> x 188)) email-ids) 177)
				 (setf id (+ 189 (random 177)))
				 (loop while (let ((h-id (+ 189 (random 177))))
					       (setf id h-id)
					       (member h-id email-ids)))))
			 id)))
	(with-model-eval user
	  ;;set goal
	  (mod-focus-fct '(current-goal class-email))
	  ;;add email info to imaginal buffer
	  (multiple-value-bind (chunk-name sender subj body link linkt)
	      (no-output (values (intern (format nil "EMAIL-~D" trial))
				 (cdr (assoc 'sender (cdr (assoc email-id *EmailData*))))
				 (cdr (assoc 'subject (cdr (assoc email-id *EmailData*))))
				 (cdr (assoc 'body (cdr (assoc email-id *EmailData*))))
				 (cdr (assoc 'link (cdr (assoc email-id *EmailData*))))
				 (cdr (assoc 'linktext (cdr (assoc email-id *EmailData*))))))
	    (format t "Participant: ~A~%Trial: ~A~%Email-ID: ~A~%Email-Type: ~A~%"
		    user trial email-id (if (< email-id 189) 'phish 'ham))
	    (define-chunks-fct (list `(,chunk-name
				       isa email
				       sender ,sender
				       subj ,subj
				       body ,body
				       link ,link
				       linkt ,linkt
				       class nil
				       conf nil
				       )))
	    (set-buffer-chunk 'imaginal chunk-name)))
	(push email-id email-list)))
    (setf email-list (nreverse email-list))
    (setf *email-list* email-list)

    ;;run model
    (run 50)
    
    (let ((mullers '()))
      ;;reset *obs-states* each trial
      (setf *obs-states* '())
      (dotimes (i 10)
	(let* ((user (nth i user-list))
	       (email-id (nth i email-list)))
	  (with-model-eval user
	    ;;save data to file (and to a-list for use with selection algorithm)
	    (multiple-value-bind (email-type sender subj body link linkt classification confidence)
		(no-output (values (cdr (assoc 'type (cdr (assoc email-id *EmailData*))))
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'sender)
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'subj)
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'body)
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'link)
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'linkt)
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'class)
				   (chunk-slot-value-fct (first (buffer-chunk imaginal)) 'conf)
				   ))
	      (let ((accuracy (if (equal email-type classification)
				  1.0
				  0.0)))
		(setf *obs-states* (acons user (if (equal email-type 'phishing)
						   (if (= accuracy 1.0)
						       1
						       0)
						   'none)
					  *obs-states*))
		;;now save the data to file
		(with-open-file (str (make-pathname :type "txt" :name "MURI_PhishingModel_MAB_v1.7_RMAB_Sims")
				     :direction :output
				     :if-exists :append
				     :if-does-not-exist :create)
	          (format str "~A~C~A~C~A~C~A~C~A~C~A~C~A~C~A~C~A~C~A~%"
			  user #\Tab phase #\Tab (1+ trial) #\Tab email-type #\Tab email-id #\Tab classification #\Tab confidence #\Tab
			  (if (equal classification 'HAM) (/ confidence 100) (/ (- 100 confidence) 100)) #\Tab
			  (if (equal classification 'PHISHING) (/ confidence 100) (/ (- 100 confidence) 100)) #\Tab accuracy
			  ))
		
		;;give feedback (default: only if click on phishing email and only during phase 2)
		;;If click on phishing email, and incorrect classification, provide feedback by updating the slots of the imaginal buffer.
		;;modify the goal so the production "get-feedback" fires if need feedback or "no-feedback" if do not need feedback.
		;;allow for mulling of phishing emails to account for frequency effects (i.e., add extra rehearsals or multiplying factor for bll
		(cond ((and (equal phase 'training) (equal email-type 'phishing) (equal accuracy 0.0))
		       (mod-buffer-chunk 'imaginal `(class ,email-type conf 100.0))
		       (mod-focus-fct '(current-goal get-feedback))
		       (setf mullers (acons user (pairlis '(sender subj body link linkt email-type)
							  (list sender subj body link linkt 'phishing))
					    mullers))
		       )
		      (t
		       (mod-focus-fct '(current-goal no-feedback))))))
	    )))

      ;;Run the model for 5s to update imaginal buffer, harvest its chunk, and add extra time to simulate getting feedback.
      (run-full-time 5.0)

      ;;mulling
      (dolist (user mullers)
	(with-model-eval (car user)
	  (let ((num-mulls (cdr (assoc (car user) *mull-list*))))
	    (if (< num-mulls *mull-lim*)
		(progn
		  (setf (cdr (assoc (car user) *mull-list*)) (incf num-mulls))
		  (dotimes (j 1) ;;mull N times
		    (merge-dm-fct `((,(intern (format nil "EMAIL-~D-~D" trial (1+ j)))
				     isa email
				     sender ,(cdr (assoc 'sender (cdr user)))
				     subj ,(cdr (assoc 'subj (cdr user)))
				     body ,(cdr (assoc 'body (cdr user)))
				     link ,(cdr (assoc 'link (cdr user)))
				     linkt ,(cdr (assoc 'linkt (cdr user)))
				     class ,(cdr (assoc 'email-type (cdr user)))
				     conf 100.0)))))))))
      ))
  )

  
;;run-task will loop through entire experiment (default 100 trials)
(defun run-task (trials runs)
  (format t "+-+-+-+-+-+-+-+-Start Model-+-+-+-+-+-+-+-+~%")
  ;;first reverse *user-list*
  (setf *user-list* (nreverse *user-list*))
  ;;Start by opening a file to write data to
  (with-open-file (str (make-pathname :type "txt" :name "MURI_PhishingModel_MAB_v1.7_RMAB_Sims")
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
    (format str "Participant~CPhase~CTrial~CEmailType~CEmailID~CClassification~CConfidence~CpHam~CpPhishing~CAccuracy~%"
	   #\Tab #\Tab #\Tab #\Tab #\Tab #\Tab #\Tab #\Tab #\Tab))

  ;;Open file to write initi data to
  (with-open-file (str (make-pathname :type "txt" :name "MURI_PhishingModel_MAB_v1.7_RMAB_Sims_InitData")
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
    (format str "Participant~CInitLength~CHamRatio~CNumHamInit~CIndex~CEmailID~CClassification~%"
	    #\Tab #\Tab #\Tab #\Tab #\Tab #\Tab))

  ;;Read in phishing data and ham data into a-list
  (cl-csv:do-csv (row (make-pathname :type "csv" :name "combinedemailsdata_cleaned_updated_20221122") :skip-first-p t)
    (setf *EmailData* (acons (read-from-string (first row))
			     (pairlis '(type sender subject body link linktext)
				      (list (read-from-string (second row)) (simplify-sender (third row)) (fourth row) (fifth row) (simplify-link (sixth row)) (simplify-link (seventh row))))
			     *EmailData*)))

  ;;Read in Email Similarity data into a-list
  (cl-csv:do-csv (row (make-pathname :type "csv" :name "EmailSimilarityMatrix_sum_decomp") :skip-first-p t)
	   (let ((email-id (read-from-string (first row))))
	     (setf *EmailSims* (acons (read-from-string (first row))
				      nil
				      *EmailSims*))
	     (dotimes (i 367)
	       (setf (cdr (assoc email-id *EmailSims*)) (acons (1+ i)
							       (read-from-string (nth (+ 1 i) row))
							       (cdr (assoc email-id *EmailSims*)))))))

  ;;Initialize models (default: with n random emails of which random x% are ham) and save data to file
  ;;run through user list and add email ids to list, only initialize on their turn
  (dolist (user *user-list*)
      (let* ((email-ids (mapcar #'car (cdr (assoc user *user-history*))))
	     (dm-count 0)
	     (num-init (nth (+ (random 7) 0) '(10 20 30 40 50 60 70 80 90 100))) ;;randomize length of initial memory (uniform distribution)
	     (ham-ratio (float (/ (* 5 (round (/ (round (* (max (min (+ 6.0 (approx-act-r-noise 0.5)) 7.5) 4.5) 10))
						 (float 5))))
				  100))) ;; randomize ratio with normal distribution, range 45%-75%, to spare enough ham emails
	     (num-ham-init (round (* ham-ratio num-init)))) ;; number of initialized ham emails 
	(dotimes (k num-init)
	  (if (< k num-ham-init)
	      ;;if k < number of ham init, add ham email
	      (let ((ham-id (let ((id nil))
			      (loop while (let ((h-id (+ 189 (random 177))))
					    (setf id h-id)
					    (member h-id email-ids)))
			      id)))
		(incf dm-count)
		;;save list of init ham-ids
		(push ham-id email-ids)		
		;;save user init info
		(save-init-data user num-init ham-ratio num-ham-init dm-count ham-id 'HAM))
	      ;;else add phishing email
	      (let ((phish-id (let ((id nil))
				(loop while (let ((p-id (+ 1 (random 188))))
					      (setf id p-id)
					      (member p-id email-ids)))
				id)))
		(incf dm-count)
		;;save list of init phish-ids
		(push phish-id email-ids)		
		;;save user init info
		(save-init-data user num-init ham-ratio num-ham-init dm-count phish-id 'PHISHING))))
	
	;;add email-ids for user to *init-list*
	(setf *init-list* (acons user email-ids *init-list*))))

  ;;set up pre-test and post-test lists
  (dolist (user *user-list*)
    (let* ((email-list '(phish ham phish ham phish ham phish ham phish ham phish ham phish ham phish ham phish ham phish ham))
	   (pre-test-list (shuffle (copy-seq email-list)))
	   (post-test-list (shuffle (copy-seq email-list))))
      (setf *pre-test-list* (acons user pre-test-list *pre-test-list*))
      (setf *post-test-list* (acons user post-test-list *post-test-list*))))

  ;;run 10 users at a time for n times
  ;;run trials
  (dotimes (k runs)
    (setf *group-id* k)
    (setf *user-history* '()) ;;reset *user-history* each run so only current users in a-list
    (let ((user-list (subseq *user-list* (* k 10) (+ 10 (* k 10)))))
      ;;initialize 10 models
      (dolist (user user-list)
	(let ((dm-count 0))
	  (with-model-eval user
	    (dolist (email-id (cdr (assoc user *init-list*)))
	      (if (> email-id 188)
		  ;;create ham email
		  (progn
		    (multiple-value-bind (sender subj body link linkt)
			(no-output (values (cdr (assoc 'sender (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'subject (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'body (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'link (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'linktext (cdr (assoc email-id *EmailData*))))))
		      (merge-dm-fct (list `(,(intern (format nil "INIT-~D" (incf dm-count)))
					    isa email
					    sender ,sender
					    subj ,subj
					    body ,body
					    link ,link
					    linkt ,linkt
					    class ham
					    conf 100.0
					    ))))
		    ;;save ham email to *user-history*; if user has user-history, append list, else create new user
	            (if (assoc user *user-history*)
			(setf (cdr (assoc user *user-history*)) (acons email-id `((CLASS . HAM) (TIME . ,(mp-time))) (cdr (assoc user *user-history*))))
			(setf *user-history* (acons user `((,email-id . ((CLASS . HAM) (TIME . ,(mp-time))))) *user-history*)))
		    )
		  ;;create phishing email
		  (progn
		    (multiple-value-bind (sender subj body link linkt)
			(no-output (values (cdr (assoc 'sender (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'subject (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'body (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'link (cdr (assoc email-id *EmailData*))))
					   (cdr (assoc 'linktext (cdr (assoc email-id *EmailData*))))))
		      (merge-dm-fct (list `(,(intern (format nil "INIT-~D" (incf dm-count)))
					    isa email
					    sender ,sender
					    subj ,subj
					    body ,body
					    link ,link
					    linkt ,linkt
					    class phishing
					    conf 100.0
					    ))))
		    ;;save phish email to *user-history*; if user has user-history, append list, else create new user
	            (if (assoc user *user-history*)
			(setf (cdr (assoc user *user-history*)) (acons email-id `((CLASS . PHISHING) (TIME . ,(mp-time))) (cdr (assoc user *user-history*))))
			(setf *user-history* (acons user `((,email-id . ((CLASS . PHISHING) (TIME . ,(mp-time))))) *user-history*)))
		    )))))
	)
      (dotimes (i trials)
	    (run-trial i k user-list))
      )
    (format t "~A~%~%+-+-+-+-+-+-+-+-End Run ~A-+-+-+-+-+-+-+-+~%" (mp-time) k)
    )
  (format t "~A~%~%+-+-+-+-+-+-+-+-End Model-+-+-+-+-+-+-+-+~%" (mp-time))
  )


;;; linear-similarity function
(defun linear-similarity (x y)
  (cond ((equal x y) 0.0)
	((or (string-equal x 'none) (string-equal y 'none)) (no-output (first (sgp :md)))) ;;if 'none or "none", then md
        ((and (stringp x) (stringp y)) (1- (ss:cached-semantic-textual-similarity x y :remote)))
        (t (no-output (first (sgp :md)))))) ;; currently at -1.0

(vom:config t :info)

(clear-all)

(dotimes (i 1000)
  (let ((user (intern (format nil "USER-~A" (1+ i)))))
    (push user *user-list*)
    (setf *mull-list* (acons user 0 *mull-list*))
    (define-model-fct user
(list
;;; Define paramenters
'(sgp :esc t
     :sim-hook linear-similarity
     :md -1.0
     :seed (987654321 3) ;;set seed to be the same for each model

     ;;Activation parameters set to defaults for now
     :ans 0.25
     :tmp nil
     :mp 2.0
     :bll 0.5 ;;0.5
     :rt -10.0 ;;set low to ensure retrieval
     :blc 5    ;;set high to ensure retrieval
     :lf 0.25  ;;set low to ensure retrieval
     :le 0.1   ;;set low to ensure retrieval
     ;;:dat 0.005 ;;set low to reduce production firing time, remove to reset to default 0.05
     :ol nil
     :ga 0.0   ;;set to 0 for no spreading activation from goal buffer
     :imaginal-activation 0.0 ;;set to 0 for no spreading activaiton from imaginal buffer

     ;; trace parameters
     :v nil
     :blt t
     :sblt t
     :trace-detail high
     ) ;;end sgp

;;; Chunk-types
'(chunk-type goal current-goal)
'(chunk-type email sender subj body link linkt class conf)

;;; Initial DM chunks
'(add-dm (read-email isa goal current-goal read-email)
        )

;;; Productions

;;Read email
'(p read-email
   =goal>
    isa goal
    current-goal read-email
 ==>
  =goal>
  current-goal nil
)

;;Classify email
'(p class-email
   =goal>
    isa goal
    current-goal class-email
   =imaginal>
    isa email
    sender =sender
    subj =subject
    body =body
    link =link
    linkt =linkt
   ?blending>
    state free
 ==>
   +blending>
    isa email
    sender =sender
    subj =subject
    body =body
    link =link
    linkt =linkt
    :do-not-generalize (class)
    :ignore-slots (conf)
   =imaginal>
   =goal>
    current-goal give-classification
)

;;Give classification for email with a link
'(p give-class
   =goal>
    isa goal
    current-goal give-classification
   =blending>
    isa email
    class =class
   =imaginal>
   !bind! =conf (* 100.0 (caddr (assoc 'class (cdar (last (confidence-values))))))
 ==>
   @blending> ;;do not save blended chunk in DM; later might get feedback and save the chunk
   =imaginal>
    class =class
    conf =conf
   =goal>
   current-goal nil
   !eval! (format t "Classification is: ~A (ham: ~A, phish: ~A)~%"
	   =class
	   (if (equal =class 'ham) (/ =conf 100) (- 1 (/ =conf 100)))
	   (if (equal =class 'phishing) (/ =conf 100) (- 1 (/ =conf 100))))
   !eval! (format t "Confidence is: ~A~%" =conf)
)

;;get-feedback
'(p get-feedback
   =goal>
    isa goal
    current-goal get-feedback
   =imaginal>
    class =class
   !bind! =current-model (current-model)
   !bind! =current-time (mp-time)
 ==>
   -imaginal> ;;save chunk in DM ;this represents the GT from feedback during phase 2
   =goal>
   ;;now save the data to a-list for use with selection algorithm
   !eval! (let ((email-id (nth (1- (- (parse-integer (string =current-model) :start 5) (* 10 *group-id*))) *email-list*)))
	    (if (assoc email-id (cdr (assoc =current-model *user-history*))) ;;if already seen email, just replace current time and classification
		(setf (cdr (assoc email-id (cdr (assoc =current-model *user-history*)))) (list (cons 'CLASS =class) (cons 'TIME =current-time)))
		(if (assoc =current-model *user-history*)
		    (setf (cdr (assoc =current-model *user-history*)) (acons email-id
									     (list (cons 'CLASS =class) (cons 'TIME =current-time)) (cdr (assoc =current-model *user-history*))))
		    (setf *user-history* (acons =current-model (list (cons email-id (list (cons 'CLASS =class) (cons 'TIME =current-time)))) *user-history*)))))
	    
)

;;no-feedback
'(p no-feedback
   =goal>
    isa goal
    current-goal no-feedback
   =imaginal>
    class =class
   !bind! =current-model (current-model)
   !bind! =current-time (mp-time)
 ==>
   -imaginal> ;;save chunk in DM ;this represents the given answer during phase 2
   =goal>
   ;;now save the data to a-list for use with selection algorithm
   !eval! (let ((email-id (nth (1- (- (parse-integer (string =current-model) :start 5) (* 10 *group-id*))) *email-list*)))
	    (if (assoc email-id (cdr (assoc =current-model *user-history*))) ;;if already seen email, just replace current time and classification
		(setf (cdr (assoc email-id (cdr (assoc =current-model *user-history*)))) (list (cons 'CLASS =class) (cons 'TIME =current-time)))
		(if (assoc =current-model *user-history*)
		    (setf (cdr (assoc =current-model *user-history*)) (acons email-id
									     (list (cons 'CLASS =class) (cons 'TIME =current-time)) (cdr (assoc =current-model *user-history*))))
		    (setf *user-history* (acons =current-model (list (cons email-id (list (cons 'CLASS =class) (cons 'TIME =current-time)))) *user-history*)))))
)


))))
