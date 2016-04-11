import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import array
from sklearn.externals import joblib
import timeit




winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
















clf = joblib.load('shane.pkl')

print type(clf)

cap = cv2.VideoCapture('../../../videoData/input_video_sample1.mov')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.BackgroundSubtractorMOG(history=500,varThreshold=20, bShadowDetection= False)
k=0

label=0
print 'Classifier Trained'

while(1):
	ret, frame = cap.read()
	height, width = frame.shape[:2]
	frame = cv2.resize(frame,(900, 700), interpolation = cv2.INTER_CUBIC)
	fgmask = fgbg.apply(frame)
	
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	
	fgamsk= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
	
	cnts, hierarchy = (cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) ) 
	
	

	mg2_fg = cv2.bitwise_and(frame,frame,mask = fgmask)
	for c in cnts:

		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 10000:
			continue
		
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

		img = cv2.cvtColor(mg2_fg, cv2.COLOR_BGR2GRAY)
	
		im=img[y:y+h,x:x+w]
	  
		cv2.imwrite('./data/1img{:>07}.jpg'.format(k),im)
		
		



   

	cv2.imshow('frame',frame)
	ret, frame = cap.read()
	ret, frame = cap.read()
	ret, frame = cap.read()
	ret, frame = cap.read()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()