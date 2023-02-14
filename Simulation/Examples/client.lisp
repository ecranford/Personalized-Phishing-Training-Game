;;; Copyright (c) 2022 Carnegie Mellon University
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

(ql:quickload '(:usocket :cl-json))

(defparameter +host+ "127.0.0.1")
(defparameter +port+ 9893)

;; DANGER, DANGER: This is pretty simple minded. It has no error handling or checking of
;;                 data for the correct format. If anything isn't just as it's expected to
;;                 be Bad Things will happen and there will be tears.

(defun make-request ()
  ;;`((:word . ,(format nil "~R" (1+ (random 20))))
  ;;  (:repeat . ,(1+ (random 10)))))
  (let ((round 1)
        (user1-action 1)
        (user2-action 0)
        (user1-reward 1)
        (user2-reward 0))
       `((:round . ,round)
         (:action . ((:user1 . ,user1-action) (:user2 . ,user2-action)))
         (:rewward . ((:user1 . ,user1-reward) (:user2 . ,user2-reward))))
       )
  )

(defun run()
  (let ((socket (usocket:socket-connect +host+ +port+)))
    (unwind-protect
         (let ((stream (usocket:socket-stream socket))
               (data (make-request)))
           (format t ";; request: ~S~%" data)
           (setf data (json:encode-json-to-string data))
           (format t ";; sending: ~S~%" data)
           (format stream "~A~%" data)
           (finish-output stream)
           (setf data (read-line stream))
           (format t ";; received: ~S~%" data)
           (format t ";; decoded response: ~S~%" (json:decode-json-from-string data)))
      (usocket:socket-close socket))))
