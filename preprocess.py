import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd 
from PIL import Image
import matplotlib.image as mpimg

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

df = pd.read_csv('fer2013/fer2013.csv')
y = 1

for i in df['pixels']:
	x = i.split()
	for j in range(len(x)):
		x[j] = int(x[j])

	x = np.array(x)
	x = x.reshape(48,48)
	im = Image.fromarray(x.astype(np.uint8))
	im.save(os.path.join(str(y) + '.png'))
	y+=1

# img = mpimg.imread('1.png')
# plt.imshow(img, cmap='gray')
# plt.show()