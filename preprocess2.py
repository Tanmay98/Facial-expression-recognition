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

image_size = 150
data_dir = "E:/DL/facial-expression/images"
df = pd.read_csv('fer2013/fer2013.csv')

X = []
y = np.array(df['emotion'])

for img in tqdm(os.listdir(data_dir)):
	img_array = cv2.imread(os.path.join(data_dir, img))
	new_array = cv2.resize(img_array, (image_size,image_size), interpolation=cv2.INTER_CUBIC)
	p = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)
	    
	# Get faces into webcam's image
	rects = detector(new_array, 0)

	# For each detected face, find the landmark.
	for (i, rect) in enumerate(rects):
	    # Make the prediction and transfom it to numpy array
	    shape = predictor(new_array, rect)
	    shape = face_utils.shape_to_np(shape)

	#for calculating euclidian distances between 68 facial points
	distances = []
	for i in shape:
		for j in range(len(shape)):
			dist = math.sqrt((shape[j][0]-i[0])**2 + (shape[j][1]-i[1])**2)
			distances.append(dist)
	    
	X.append(distances)


pickle_out = open("X_pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# Thanks guys, I figured it out, I was supposed to add a try/exception so my code could bypass "ugly" images:

#     try:
#     path=os.path.join(mypath,n)
#         img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img=cv2.resize(img, (img_rows,img_cols))

#     except Exception as e:
#         print(str(e))
