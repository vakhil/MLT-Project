import json
import cv2
import numpy as np
import os
import ruamel.yaml as yaml

content_list = []
frames  = {}

with open('input_video_sample1.json') as json_data:
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

print (frames['675'])

lists = {}
count = 0
while count < len(frames): 
	#print "./images/frame%s"%count
	path = "./images/frame"+str(count)
	path = path+'.jpg'
	print path
	img = cv2.imread(path)	
	#print img
	counts = str(count)
	if counts in frames :
		tames = frames[counts]	
		print counts
		for items in tames :
			
			if items[0] == 1:

				continue
			else :
				crop_img = img[int(items[2]):int(items[-2]) , int(items[1]):int(items[3])].copy()
				if items[-1] in lists :
					lists[items[-1]] += 1
				else :
					lists[items[-1]] = 1
				var = "./finale/input_1/"+str(items[-1])+"/input1_"+str(items[-1])+str(lists[items[-1]])+str('.jpg')
				
				cv2.imwrite(var, crop_img)
	
	count = (int(count) + 1)


