import json
import cv2
import numpy as np
import os
import ruamel.yaml as yaml

content_list = []
frame_dict  = {}



cap = cv2.VideoCapture("../../../videoData/datasample1.mov")
#success,image = cap.read()
fgbg = cv2.BackgroundSubtractorMOG2()

while True:
		ret, frame = cap.read() 
		print (frame.shape)   
		fgmask = fgbg.apply(frame) 
		contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
		i = 0  
		idx =0 
		for cnt in contours:
		    idx += 1
		    x,y,w,h = cv2.boundingRect(cnt)
		    print x,y,w,h
		    roi=frame[y:y+h,x:x+w]
		    cv2.imwrite('./dick/'+str(idx) + '.jpg', roi)
		    #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
		   
				
			   
		   
		cv2.imshow('frame',fgmask)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	
cap.release()
cv2.destroyAllWindows()

