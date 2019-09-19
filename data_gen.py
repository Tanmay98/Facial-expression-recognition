import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array, load_img
import os
import cv2
import numpy as np
from tqdm import tqdm

train_datagen=ImageDataGenerator(rotation_range=10,
                           width_shift_range=4,
                           height_shift_range=4,
                           rescale=1/255.0,
                           horizontal_flip=False,
                           vertical_flip=False,
                           fill_mode='nearest')

# labels = pickle.load(open("labels", "rb"))
# labels = np.array(labels)
# h = []
# for j in range(len(labels)):
#   if labels[j] == 3:
#     h.append(j)

image_folder = "E:/DL/facial-expression/images"
files=sorted(os.listdir(image_folder))
files=list(map(lambda x: os.path.join(image_folder,x),files))
a=(len(files))
empty = []
for i in tqdm(range(a)):
    im = (files[i])
    s=im[31:]
    p=s[:-4]
    img=cv2.imread(im) 
    img = img.reshape((1,) + img.shape)
    i = 0
    for batch in train_datagen.flow(img, batch_size=1,save_to_dir='augmented_images', save_prefix=p, save_format='png'):
        #print(batch.shape)
        i += 1
        if i >5 : ## making 10
            break  # otherwise the generator would loop indefinitely and generate indefinite images

