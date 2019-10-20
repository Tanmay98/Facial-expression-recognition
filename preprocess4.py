#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import dlib
from imutils import face_utils 
from tqdm import tqdm
import math
import pickle
from random import shuffle
from mtcnn.mtcnn import MTCNN 

image_size = 150
data_dir = "E:/DL/facial-expression/generated-images"

X = np.zeros(829707440).reshape(179435,4624)

name = 0
temp = 0 
b = 0

#this is to identify some of the missing generated images in our generated images folder
list_img = []
for images in os.listdir(data_dir):
	image = os.path.join(data_dir, images)
	r = image[41:]
	meh = r.split('_')
	name = int(meh[0])
	list_img.append(name)
	
listy = [i+1 for i in range(35887)]
missing = []
for i in listy:
	if list_img.count(i) < 6:
		missing.append(i)

#balancing the generated images dataset
for images in tqdm(os.listdir(data_dir)):
	image = os.path.join(data_dir, images)
	r = image[41:]
	meh = r.split('_')
	name = int(meh[0])
	name2 = int(meh[1][:-4])

	flag = 0
	
	for i in missing:
		if i == name:
			flag = 1

	if flag == 0:
		if name2 == 1:
			os.remove(image)

#preparing our X 
detector = MTCNN()
tempo = 1
for img in tqdm(os.listdir(data_dir)):
	image = os.path.join(data_dir, img)
	img_array = cv2.imread(image)
	new_array = cv2.resize(img_array, (image_size,image_size), interpolation=cv2.INTER_CUBIC)
	
	rects_list = []

	p = "shape_predictor_68_face_landmarks.dat"
	# detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)
	result = detector.detect_faces(new_array)

	if result:
		rects = result[0]['box']
		face_location = dlib.rectangle(rects[0], rects[1], rects[2], rects[3])
		rects_list.append(face_location)
		
		# plotting the image with its face detected
		cv2.rectangle(new_array,
	              (rects[0], rects[1]),
	              (rects[0]+rects[2], rects[1] + rects[3]),
	               (0,155,255),
	              2)
		plt.imshow(new_array, cmap='gray')
		plt.show()	

		# For each detected face, find the landmark.
		for (i, rect) in enumerate(rects_list):
		    # Make the prediction and transfom it to numpy array
		    shape = predictor(new_array, rect)
		    shape = face_utils.shape_to_np(shape)

		#for calculating euclidian distances between 68 facial points
		distances = []
		for i in shape:
			for j in range(len(shape)):
				dist = math.sqrt((shape[j][0]-i[0])**2 + (shape[j][1]-i[1])**2)
				distances.append(dist)
	else:
		distances = []
		distances.append('NaN')

	# assigning distance value to the proper index in X numpy array
	r = image[41:]
	meh = r.split('_')
	name = int(meh[0])

	if temp != name:
		b = 0

	temp = name
	X[(name-1)*5 + b] = distances # a +(n-1)d can be thought as of ap series which gives the general term and we can fetch easily each random term
	b+=1

pickle_out = open("X_pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()


