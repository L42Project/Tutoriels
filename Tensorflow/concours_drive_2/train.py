import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
import random
import cv2
import model
import traitement_images as ti

dir_images='./training/images/'
dir_mask  ='./training/1st_manual/'

if not os.path.isdir(dir_images):
    quit("The directory {} don't exist !".format(dir_images))
if not os.path.isdir(dir_mask):
    quit("The directory {} don't exist !".format(dir_mask))

tab_images=[]
tab_masks=[]

list_file=os.listdir(dir_images)
if list_file is None:
    quit("No file in {} !".format(dir_images))
    
for fichier in list_file:
    img_orig=cv2.imread(dir_images+fichier)
    tab_images.append(img_orig[:576, :560])
    num=fichier.split('_')[0]
    file_mask=dir_mask+num+'_manual1.gif'
    if not os.path.isfile(file_mask):
        quit("Mask of {} don't exist in {}".format(file_mask, dir_mask))
    img_mask_orig=np.array(Image.open(file_mask))
    tab_masks.append(img_mask_orig[:576, :560])

    for angle in range(0, 360, 30):
        img_r=ti.rotateImage(img_orig, angle)
        img=img_r.copy()
        img=ti.random_change(img)
        tab_images.append(img[:576, :560])
        img_mask=ti.rotateImage(img_mask_orig, angle)
        tab_masks.append(img_mask[:576, :560])
        
        img=cv2.flip(img_r, 0)
        img=ti.random_change(img)
        tab_images.append(img[:576, :560])
        img_m=cv2.flip(img_mask, 0)
        tab_masks.append(img_m[:576, :560])

        img=cv2.flip(img_r, 1)
        img=ti.random_change(img)
        tab_images.append(img[:576, :560])
        img_m=cv2.flip(img_mask, 1)
        tab_masks.append(img_m[:576, :560])

        img=cv2.flip(img_r, -1)
        img=ti.random_change(img)
        tab_images.append(img[:576, :560])
        img_m=cv2.flip(img_mask, -1)
        tab_masks.append(img_m[:576, :560])

tab_images=np.array(tab_images, dtype=np.float32)/255
tab_masks =np.array(tab_masks,  dtype=np.float32)[:, :, :]/255

train_images, test_images, train_masks, test_masks=train_test_split(tab_images, tab_masks, test_size=0.05)

del tab_images
del tab_masks

my_model=model.model(64)

my_model.compile(optimizer='adam',
                 loss=model.LossDice,
                 metrics=['accuracy'])
my_model.fit(train_images,
             train_masks,
             epochs=20,
             batch_size=4,
             validation_data=(test_images, test_masks))

dir_test_images='./test/images/'

tab_test_images=[]
tab_files=[]
for fichier in os.listdir(dir_test_images):
    img=cv2.imread(dir_test_images+fichier)
    tab_test_images.append(img[:576, :560])
    tab_files.append(fichier.split('_')[0])

tab_test_images=np.array(tab_test_images, dtype=np.float32)/255
tab_files=np.array(tab_files)

for id in range(len(tab_test_images)):
    mask=np.zeros((584, 565, 1), dtype=np.float32)
    prediction=my_model.predict(np.array([tab_test_images[id]]))
    mask[:576, :560]=prediction[0]*255
    cv2.imwrite("./predictions/"+str(tab_files[id])+".png", mask)

