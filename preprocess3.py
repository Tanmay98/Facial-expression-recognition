import os
import pickle 
import numpy as np 
from tqdm import tqdm

data_dir = "E:/DL/facial-expression/images"
augmented_data_dir = "E:/DL/facial-expression/augmented_images"

labels = pickle.load(open("labels", "rb"))
labels = np.array(labels)

h = []
labels_balanced = []

for j in range(len(labels)):
  if labels[j] == 3:
    h.append(int(j) + 1)

for num in tqdm(h):	
	for images in os.listdir(augmented_data_dir):
		image = os.path.join(augmented_data_dir, images)
		s = image[41:]
		p = s.split('_')
		pp = int(p[0])
		if pp == h[1]:
			os.remove(image)
		


#making labels or y for augmented images
y = []
for imges in os.listdir(augmented_data_dir):
	imge = os.path.join(augmented_data_dir, imges)
	r = imge[41:]
	meh = r.split('_')
	name = int(meh[0])
	y.append(labels[name])
