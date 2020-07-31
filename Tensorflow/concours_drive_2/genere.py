import tensorflow as tf
from PIL import Image
import os
import numpy as np
import model
import cv2

dir_test_images='./test/images/'

import sys
np.set_printoptions(threshold=sys.maxsize)

tab_test_images=[]
tab_files=[]
for fichier in os.listdir(dir_test_images):
    img=np.array(Image.open(dir_test_images+fichier))
    tab_test_images.append(img[:576, :560])
    tab_files.append(fichier.split('_')[0])
    
tab_test_images=np.array(tab_test_images, dtype=np.float32)/255
tab_files=np.array(tab_files)

my_model=tf.keras.models.load_model('saved_model/my_model', custom_objects={'loss': model.dice_loss()}) #, custom_objects={'dice_loss': dice_loss})

for id in range(len(tab_test_images)):
    mask=np.zeros((584, 565, 1), dtype=np.float32)
    prediction=my_model.predict(np.array([tab_test_images[id]]))
    mask[:576, :560]=prediction[0]*255
    #mask[mask<150]=0
    #mask[mask>200]=255
    #print(mask)
    cv2.imwrite("./predictions/"+str(tab_files[id])+".png", mask)

