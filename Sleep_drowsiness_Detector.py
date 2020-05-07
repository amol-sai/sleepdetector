# import the necessary packages
# Dlib : If you dont have dlib installed i suggested you to please steps in Dlib_Installation document,
# Dlib installation can sometime be tricky and i strgonly suggest that you use Python 3.6 to install dlib
# as on more latest version Dlib installation is just complicated and not worth spending time for this trial

import dlib
import emoji 
import playsound
import argparse
import imutils
import time
import numpy as np  ## numpy is essential 
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

## Function Defination
#This function is used to compute the ratio of distances 
#between the vertical eye landmarks and the distances between the horizontal eye landmarks
# Return value of this will have 3 interpretation
#1) if eye is closed then eye aspect ration will be constant
#2) if eye is appriximately constant when eye is open
#3) if eye is blinking value will rapdily decrease towards almost zero during a blink 
# Ref: Based on Soukupová and Čech’s 2016

def eye_aspect_ratio(eye):
	# Here we are computing the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
    return ear

# Alarm sound
def sound_alarm(path):
		playsound.playsound(path)

 
Eye_AR_Threshold = 0.3 ## Eye aspect ration to indicate blink 
Consec_Frames = 31  ### Alarm will sound if eyes were closed consecutively for 31 frames

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
Counter_Value = 0
Set_Alarm = False

# argument parse and parse the arguments , this will help us to run this from command line
# in efficient way
        
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="This is our path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",help="This is path for alarm file .wav")
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("Loading.........Facial landmark predictor:")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("Hold on tight......starting video stream :")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# Here we grab the frame from the threaded video file stream to resize and 
	# and convert it to grayscale
	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# find  eft and right eye coordinates & then we will use the
		# coordinates to compute the eye aspect ratio for both eyes
        
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        # ref: https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
        
		leftHull = cv2.convexHull(leftEye)
		rightHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame Counter_Value
		if ear < Eye_AR_Threshold:
			Counter_Value += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if Counter_Value >= Consec_Frames:
				# if the alarm is not on, turn it on
				if not Set_Alarm:
					Set_Alarm = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "SLEEP ALERT", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# reset counter if eye aspect ratio is not below the blink
		
		else:
			Counter_Value = 0
			Set_Alarm = False

		
		cv2.putText(frame, "Eye Aspect Ratio : {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# break from the loop key q 
	if key == ord("q"):
        	break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print('zzz')
print('Still Sleepy ....')
print('<<<<<<<< A XL Coffee may be good option >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(emoji.emojize('Happy Learning :thumbs_up:'))