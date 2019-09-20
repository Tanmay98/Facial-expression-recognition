import os
import pickle 
import numpy as np 
from tqdm import tqdm

data_dir = "E:/DL/facial-expression/images"
augmented_data_dir = "E:/DL/facial-expression/augmented_images"

labels = pickle.load(open("labels", "rb"))
labels = np.array(labels)

#balancing out happy images in the dir
h = []
labels_balanced = []

for j in range(len(labels)):
  if labels[j] == 3:
    h.append(int(j) + 1)

for num in tqdm(h):	
	temp = 0
	for images in os.listdir(augmented_data_dir):
		image = os.path.join(augmented_data_dir, images)
		s = image[41:]
		p = s.split('_')
		pp = int(p[0])
		if pp == num:
			temp += 1
			os.remove(image)
		if temp >= 2:
			break
				


#making labels or y for augmented images
y = np.zeros(35887).reshape(35887,1)

for imges in os.listdir(augmented_data_dir):
	imge = os.path.join(augmented_data_dir, imges)
	r = imge[41:]
	meh = r.split('_')
	name = int(meh[0])
	y[name - 1] = int(labels[name - 1])


for element in range(len(y)):
	y[element] = int(y[element])

pickle_out = open("balanced-labels", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
