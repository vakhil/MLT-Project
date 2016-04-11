import json
import cv2
import numpy as np
import os
import ruamel.yaml as yaml
from numpy import array
from skimage.feature import hog 
from sklearn.svm import LinearSVC
import pickle
from sklearn.externals import joblib
import os


content_list = []
frames  = {}

with open('input_video_sample3.json') as json_data:
		#json_data.encode('ascii', 'ignore')
		#d = json.load(json_data)  
		d = yaml.safe_load(json_data)
			
		for item in d:
			label = d[item]["label"]
			for frame in d[item]["boxes"]:
				
				if not frame in frames:
					frames[frame] = []
				
				temp_list = [d[item]["boxes"][frame]["outside"],d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
				frames[frame].append(temp_list)



#For sacing images to file --- the initial step *******************************
# cap = cv2.VideoCapture('../../../videoData/input_video_sample3.mov')
# fcount = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
# count = 0

# while(cap.isOpened()):
	
# 	success, image = cap.read()
# 	if success:
# 			strs = './images/frame' + str(count)+'.jpg'
# 			print strs
# 			cv2.imwrite(strs, image)
# 			print count
# 			count += 1
# 			#cv2.imshow('frame',image)
			
# 	else:
# 			break

	
# cv2.destroyAllWindows()
# cap.release()
#************************



print frames['600']

lists = {}
count = 0
print len(frames)
while count < 1079: 
	#print "./images/frame%s"%count
	path = "./images/frame"+str(count)
	path = path+'.jpg'
	
	img = cv2.imread(path)	
	#print img
	counts = str(count)
	if counts in frames :

		tames = frames[counts]	
		
		for items in tames :
			
			
			if items[0] == 1:

				continue
			else :
				crop_img = img[int(items[2]):int(items[-2]) , int(items[1]):int(items[3])].copy()
				if items[-1] in lists :
					lists[items[-1]] += 1
				else :
					lists[items[-1]] = 1
				var = "./finale/input_3/"+str(items[-1])+"/input3_"+str(items[-1])+str(lists[items[-1]])+str('.jpg')
				
				cv2.imwrite(var, crop_img)

	
 	count = (int(count) + 1)
