import pandas as pd 
import os 
import dlib
from imutils import face_utils 
from tqdm import tqdm
import math
import pickle
import numpy as np 

data_dir = "E:/DL/facial-expression/images"
df = pd.read_csv('fer2013/fer2013.csv')

labels = np.array(df['emotion'])

for n, i in enumerate(labels):
	if i == 1:
		labels[n] = 0

pickle_out = open("labels", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

# 1 = 547 0 = 4953 2 = 5121 3 = 8989 4 = 6077 5 = 4002 6 = 6198