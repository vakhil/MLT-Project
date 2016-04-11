import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import array
import os
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
















clf = joblib.load('LinearSVC.pkl')

print type(clf)
dirname = 'temp_bgsegm'
out_dirname = 'real_bgsegm'
cap = cv2.VideoCapture('../../../videoData/input_video_sample1.mov')
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5),(3,3))
fgbg = cv2.BackgroundSubtractorMOG2(history=500,varThreshold=19, bShadowDetection= False)
k=0
label=0
ff = None 
while(1):
	ret, frame = cap.read()

	height, width = frame.shape[:2]
	frame = cv2.resize(frame,(900, 700), interpolation = cv2.INTER_CUBIC)
	fgmask = fgbg.apply(frame)
	#fgmask = fgbg.apply(frame,learningRate=0.5)
	#fg1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	
	fgamsk= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
	#fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
	if ff == None :
		ff = frame
	cnts,u = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	
	mg2_fg = cv2.bitwise_and(frame,frame,mask = fgmask)
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 8000:
			continue
		
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

		img = cv2.cvtColor(mg2_fg, cv2.COLOR_BGR2GRAY)
	
		im=img[y:y+h,x:x+w]
	  
		
		
		col,row=im.shape
		Y=np.zeros((col,row),dtype=np.uint8)
		for a in range(0,col):
			for b in range(0,row):
				Y[a][b]=im[a,b]
		Z1=hog.compute(Y,winStride,padding,locations)#
		lists = []				
					
		for i in range(0, len(Z1)) :
			
			lists.append( Z1[i][0])
		finale = []
		finale.append(lists)










		
	   
		label=clf.predict(finale)
		
		
		
		if label[0]==0:
			cv2.putText(frame,"Two-Wheeler",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
		elif label[0]==1:
			cv2.putText(frame,"Three-Wheeler",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
		elif label[0]==2:
			cv2.putText(frame,"Person",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
		
	


   

	cv2.imshow('frame',frame)
	ret, frame = cap.read()
	ret, frame = cap.read()
	ret, frame = cap.read()
	ret, frame = cap.read()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()



cap.release()
cv2.destroyAllWindows()