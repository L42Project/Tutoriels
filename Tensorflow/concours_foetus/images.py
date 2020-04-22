import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import common
import config
import model

images, labels=common.prepare_data('training_set.csv')
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)
index=np.random.permutation(len(images))
images=images[index].reshape(-1, config.hauteur, config.largeur, 1)
labels=labels[index]

print("Nombre d'image:", len(images))

for i in range(len(images)):
    x, y, grand_axe, petit_axe, angle=labels[i]
    print("Label:", labels[i], angle*180)
    img_couleur=np.tile(images[i], (1, 1, 3))
    cv2.ellipse(img_couleur, (int(x*config.norm), int(y*config.norm)), (int(petit_axe*config.norm/2), int(grand_axe*config.norm/2)), angle*180, 0., 360., (0, 0, 255), 2)
    cv2.imshow("Image", img_couleur)
    
    key=cv2.waitKey()&0xFF
    if key==ord('q'):
      quit()

