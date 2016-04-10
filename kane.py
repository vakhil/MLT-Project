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

content_list = []
frames  = {}

# with open('input_video_sample1.json') as json_data:
# 		#json_data.encode('ascii', 'ignore')
# 		#d = json.load(json_data)  
# 		d = yaml.safe_load(json_data)
		    
# 		for item in d:
# 			label = d[item]["label"]
# 			for frame in d[item]["boxes"]:
				
# 				if not frame in frames:
# 					frames[frame] = []
				
# 				temp_list = [d[item]["boxes"][frame]["outside"],d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
# 				frames[frame].append(temp_list)

# print (frames['675'])

# lists = {}
# count = 0
# while count < len(frames): 
# 	#print "./images/frame%s"%count
# 	path = "./images/frame"+str(count)
# 	path = path+'.jpg'
# 	print path
# 	img = cv2.imread(path)	
# 	#print img
# 	counts = str(count)
# 	if counts in frames :
# 		tames = frames[counts]	
# 		print counts
# 		for items in tames :
			
# 			if items[0] == 1:

# 				continue
# 			else :
# 				crop_img = img[int(items[2]):int(items[-2]) , int(items[1]):int(items[3])].copy()
# 				if items[-1] in lists :
# 					lists[items[-1]] += 1
# 				else :
# 					lists[items[-1]] = 1
# 				var = "./finale/input_1/"+str(items[-1])+"/input1_"+str(items[-1])+str(lists[items[-1]])+str('.jpg')
				
# 				cv2.imwrite(var, crop_img)
	
# 	count = (int(count) + 1)



labels = []
train = []
names = { }

def training(path,train,labels):
	tame = 0
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


	for root, dir_names, file_names in os.walk(path):
		
		for path in dir_names:
			training(os.path.join(root, path),train,labels)
		for file_name in file_names:
			file_path = os.path.join(root, file_name)
			
			if not file_path in names and file_name[-3:-1] =='jp': 
				if file_name[-3:-1] == 'jp' :
					#print file_name[7:-4]
					
					if file_name[7] == 'B':
						labels.append(1)
					if file_name[7] == 'C' :
						labels.append(3)
					if file_name[7] == 'P':
						labels.append(4)
					if file_name[7] == 'N' :
						labels.append(5)
					if file_name[7] == 'M' :
						labels.append(2)


					#print file_path
					image = cv2.imread(file_path,0)
					
					
					lists = []
					
					hist = hog.compute(image,winStride,padding,locations)
					
					for i in range(0, len(hist)) :
						
						lists.append( hist[i][0])

					train.append(lists)
					tame = tame + 1
					
					#fd =hog(c, orientations=9, pixels_per_cell=(len(image)/size, (len(image[0]))/size), cells_per_block=(1, 1), visualise=False)
					#for i in len(image) :

					#train.append(fd)
					#print (fd)
					
		return tame



hell = training('./finale/input_1/',train,labels)
image = cv2.imread('gendu.jpg',0)
clf = LinearSVC()
print len(train)
print len(labels)

train = array(train)
tame = 0
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
hist = hog.compute(image,winStride,padding,locations)



lists = []					
					
for i in range(0, len(hist)) :
	
	lists.append( hist[i][0])
finale = []
finale.append(lists)
clf.fit(train,labels)
# filename = '/digits_classifier.joblib.pkl'
# _ = joblib.dump(clf, filename, compress=9)
with open('kane.pkl', 'wb') as f:
    pickle.dump(clf, f)
print clf.predict(finale), "HAKe"
#h = hog.compute(image)
#print fd.shape
# print image.shape


#image = cv2.imread("test.jpg",0)
#hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)