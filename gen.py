##generating images 

import os
import pickle 
import numpy as np 
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt 

labels = pickle.load(open("labels", "rb"))
labels = np.array(labels)

happy_indexes = []

for n,i in enumerate(labels):
	if i == 3:
		happy_indexes.append(n)

data_dir = "E:/DL/facial-expression/images"
augmented_data_dir = "E:/DL/facial-expression/generated-images"

#augmenting functions

def vert_flip(image):
    img2 = np.fliplr(image)
    return img2

def hor_flip(image):
    img2 = np.flipud(image)
    return img2

def rotate90_clock(image):
    img2 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return img2

def rotate90_anticlock(image):
    img2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img2

def increase_brightness(img, value=40):
	img += value
	img = np.clip(img, 0, 255)
	return img

#helper function for checking happy index 

def check_element(value, samplelist):
	for i in samplelist:
		if i == value:
			return True
		else:
			return False

#augmenting starts from here

choices = [0,1,2,3,4]
genfuncs = {0:vert_flip, 1:hor_flip, 2:rotate90_clock, 3:rotate90_anticlock, 4:increase_brightness}
generated_images = []

for images in tqdm(os.listdir(data_dir)):
	image = os.path.join(data_dir, images)
	img = cv2.imread(image)
	counter = int(image.split('/')[-1][7:].split('.')[0]) - 1

	temp = check_element(counter, happy_indexes)

	if temp:
		#selecting random augment functions for happy class
		for i in range(3):
			filename = augmented_data_dir + '/' + str(counter) + '_' + str(i) + '.jpg'
			selected_choice = random.choice(choices)
			ran_func_selected = genfuncs[selected_choice]
			gen_img = ran_func_selected(img)
			cv2.imwrite(filename, gen_img)
			
	else:	
		#selecting each augment function once 
		for i in range(5):
			filename = augmented_data_dir + '/' + str(counter) + '_' + str(i) + '.jpg'
			selected_choice = choices[i]
			ran_func_selected = genfuncs[selected_choice]
			gen_img = ran_func_selected(img)
			cv2.imwrite(filename, gen_img)

##generating labels

gen_labels = []

for i in labels:
	if i == 3:
		for j in range(3):
			gen_labels.append(i)
	else:
		for k in range(5):
			gen_labels.append(i)

gen_labels = np.array(gen_labels)

