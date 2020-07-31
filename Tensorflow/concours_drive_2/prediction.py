import tensorflow as tf
from PIL import Image
import os
import numpy as np
import model
import cv2

dir_test_images='./test/images/'

tab_test_images=[]
for fichier in os.listdir(dir_test_images):
    img=np.array(Image.open(dir_test_images+fichier))
    tab_test_images.append(img[:576, :560])
tab_test_images=np.array(tab_test_images, dtype=np.float32)/255

model=tf.keras.models.load_model('saved_model/my_model')

for id in range(len(tab_test_images)):
    prediction=model.predict(np.array([tab_test_images[id]]))
    cv2.imshow("image", tab_test_images[id])
    cv2.imshow("prediction", prediction[0])
    print(prediction[0])
    if cv2.waitKey()&0xFF==ord('q'):
        quit()
