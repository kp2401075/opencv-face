import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
	# print(labels)

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4,minSize=(30,30), 
				flags=cv2.CASCADE_SCALE_IMAGE)
	for (x, y, w, h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
		roi_gray = cv2.resize(roi_gray, 
					  (300, 300), 
					  interpolation=cv2.INTER_LANCZOS4)
		cv2.imshow("roi_gray",roi_gray)
		roi_color = frame[y:y+h, x:x+w]

		# recognize? deep learned model predict keras tensorflow pytorch scikit learn
		id_, conf = recognizer.predict(roi_gray)
		# print(id_)
		print(conf)
		if conf >1000 and conf < 3500:
			#print(5: #id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_TRIPLEX
			name = labels[id_]
			# print(name)
			color = (0, 0, 0)
			stroke = 2
			cv2.rectangle(frame, (x, y), (x +w, y-30), (0,255,255), -1)
			cv2.putText(frame, name, (x+5,y-10), font, 0.7, color, stroke, 2)
		else:
			cv2.rectangle(frame, (x, y), (x +w, y-30), (0,255,255), -1)
			cv2.putText(frame, "UNKNOWN", (x+5,y-10), font, 0.7, (0,0,0), stroke, 2)

		img_item = "7.png"
		cv2.imwrite(img_item, roi_color)

		color = (0, 250, 255) #BGR 0-255
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
		#subitems = smile_cascade.detectMultiScale(roi_gray)
		#for (ex,ey,ew,eh) in subitems:
		#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
