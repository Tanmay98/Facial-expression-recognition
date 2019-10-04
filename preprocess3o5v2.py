import cv2
import numpy as np 
import os 
from tqdm import tqdm
import pickle

data_dir = "E:/DL/facial-expression/generated-images"
temp = 0
temp2 = 0
temp_list = []

for images in tqdm(os.listdir(data_dir)):
	src = os.path.join(data_dir, images)
	im = src[41:]
	im_break = im.split('_')
	name = int(im_break[0])
	temp_list.append(name)

	if temp2 > 5:
		temp2 = 0

	if temp_list[temp] == name:
		if temp2 == 0:
			im_name = str(name) + "_1.png"
			dst = os.path.join(data_dir, im_name)
			os.rename(src, dst)
			temp2 +=1

		elif temp2 == 1:
			im_name = str(name) + "_2.png"
			dst = os.path.join(data_dir, im_name)
			os.rename(src, dst)
			temp2 +=1

		elif temp2 == 2:
			im_name = str(name) + "_3.png"
			dst = os.path.join(data_dir, im_name)
			os.rename(src, dst)
			temp2 +=1

		elif temp2 == 3:
			im_name = str(name) + "_4.png"
			dst = os.path.join(data_dir, im_name)
			os.rename(src, dst)
			temp2 +=1

		elif temp2 == 4:
			im_name = str(name) + "_5.png"
			dst = os.path.join(data_dir, im_name)
			os.rename(src, dst)
			temp2 +=1

		elif temp2 == 5:
			im_name = str(name) + "_6.png"
			dst = os.path.join(data_dir, im_name)
			os.rename(src, dst)
			temp2 +=1

	temp += 1
	
