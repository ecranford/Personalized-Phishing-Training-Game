(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload '(:alexandria :iterate :cl-ppcre :cl-csv :usocket-server :external-program :vom :unix-opts)))

(vom:config t :info)

(defpackage :sim-server
  (:nicknames :ss)
  (:use :common-lisp :alexandria :iterate :ppcre :usocket)
  (:export #:cached-semantic-textual-similarity
           #:load-semantic-textual-similarity-cache
           #:populate-semantic-textual-similarity-cache
           #:remote-semantic-textual-similarity
           #:run-similarity-server
           #:save-semantic-textual-similarity-cache
           #:semantic-textual-similarity
           #:make-executable))

(in-package :sim-server)

(define-constant +version+ "2021-04-16-001" :test #'equalp)

(define-constant +default-host+ "koalemos.psy.cmu.edu" :test #'equal)
(define-constant +default-port+ 9543)
(define-constant +executable-name+ "similarity-server" :test #'equal)
(define-constant +default-similarity-file+
    (make-pathname :name "similarity-cache" :type "lisp")
  :test #'equal)
(define-constant +default-mail-file+
    (make-pathname :name "emails" :type "csv")
  :test #'equal)

(defparameter *similarity-engine* nil)
(defparameter *interned-text-table* nil)
(defparameter *semantic-textual-similarity-cache* nil)

(defun clean (s)
  (unless *interned-text-table*
    (setf *interned-text-table* (make-hash-table :test 'equalp)))
  (let ((result (string-trim " " (regex-replace-all " +"
                                                    (regex-replace-all "[^ -~]+"
                                                                       (string s)
                                                                       " ")
                                                    " "))))
    (or (gethash result *interned-text-table*)
        (setf (gethash result *interned-text-table*) result))))

(define-modify-macro cleanf () clean)

(defun emit-for-similarity (stream string-1 string-2)
  (format stream "~A~%~A~%" (clean string-1) (clean string-2))
  (finish-output stream))

(defun semantic-textual-similarity (string-1 string-2)
  (unless *similarity-engine*
    (setf *similarity-engine*
          (external-program:start "java"
                                  '("-Xmx800000000000" "-cp" "joda-time.jar:xom.jar:similarity.jar:." "SimilarityMain")
                                  :input :stream
                                  :output :stream
                                  :error t))
    (iter (until (equalp (read-line (external-program:process-output-stream *similarity-engine*)) "--ready--")))
    (vom:info "Started similarity engine ~A" (external-program:process-id *similarity-engine*)))
  (handler-case
      (progn
        (emit-for-similarity (external-program:process-input-stream *similarity-engine*)
                             string-1 string-2)
        (let ((result (read (external-program:process-output-stream *similarity-engine*))))
          (vom:debug "STS of '~A' and '~A' = ~F" string-1 string-2 result)
          result))
    (error (e)
      (error "Error caught when computing semantic-textual-similarity of '~A' and '~A':~%~A"
             string-1 string-2 e))))

(defun remote-semantic-textual-similarity (string-1 string-2
                                           &key (host +default-host+) (port +default-port+))
  (with-client-socket (socket stream host port :timeout 10)
    (emit-for-similarity stream string-1 string-2)
    (read stream)))

(defun make-similarity-key (string-1 string-2)
  ;; we assume semantic-textual-similarity is commutative
  (sort (list (clean string-1) (clean string-2)) #'string-lessp))

(defun make-semantic-textual-similarity-cache ()
  (setf *semantic-textual-similarity-cache* (make-hash-table :test 'equalp)))

(defun cached-semantic-textual-similarity (string-1 string-2 &optional remote)
  (unless *semantic-textual-similarity-cache*
    (make-semantic-textual-similarity-cache))
  (let ((key (make-similarity-key string-1 string-2)))
    (or (gethash key *semantic-textual-similarity-cache*)
        (setf (gethash key *semantic-textual-similarity-cache*)
              (apply (if remote
                         #'remote-semantic-textual-similarity
                         #'semantic-textual-similarity)
                     key)))))

(defun save-semantic-textual-similarity-cache (&optional (file +default-similarity-file+))
  (with-output-to-file (s file :if-exists :supersede)
    (let ((*print-circle* t))
      (format s "~@:W~%" (hash-table-alist *semantic-textual-similarity-cache*)))))

(defun load-semantic-textual-similarity-cache (&optional (file +default-similarity-file+))
  (setf *interned-text-table* nil)
  (make-semantic-textual-similarity-cache)
  (iter (for ((s1 s2) . v) :in (with-input-from-file (s file) (read s)))
        (setf (gethash (make-similarity-key s1 s2)  *semantic-textual-similarity-cache*) v))
  file)

(defun populate-semantic-textual-similarity-cache (&key
                                                     (mail-file +default-mail-file+)
                                                     (cache-file +default-similarity-file+)
                                                     remote)
  (iter (with string-lists := (iter (for (nil nil nil . list) :in-csv (pathname mail-file))
                                    (if-first-time
                                      nil ; skip the header row
                                      (collect list))))
        (for sub-list :on string-lists)
        (for first := (first sub-list))
        (iter (for another :in (rest sub-list))
              (iter (for f :in first)
                    (for a :in another)
                    (cached-semantic-textual-similarity f a remote))))
  (save-semantic-textual-similarity-cache cache-file))

(defun process-request (stream)
  (vom:debug "request received")
  (handler-case
      (let* ((line-1 (read-line stream)) (line-2 (read-line stream)))
        (vom:debug "read: ~S, ~S" line-1 line-2)
        (let ((result (semantic-textual-similarity line-1 line-2)))
          (vom:debug "computed: ~S" result)
          (format stream "~S~%" result)
          (finish-output stream)))
    (error (e) (vom:error "Error processing request: ~A" e))))

(defun run-similarity-server (&key (port +default-port+))
  (socket-server nil port #'process-request))

(defun main ()
  (with-simple-restart (abort "Exit application")
    (handler-case (run-similarity-server)
      (error (e)
        (vom:error "Error: ~A" e)
        (opts:exit 1)))))

(defun make-executable (&optional (name +executable-name+))
  #+ccl
  (cl-user::save-application name :toplevel-function #'main :prepend-kernel t)
  #+sbcl
  (sb-ext:save-lisp-and-die name :toplevel #'main :executable t)
  #-(or ccl sbcl)
  (error "Currently executables can only be saved for CCL or SBCL"))
