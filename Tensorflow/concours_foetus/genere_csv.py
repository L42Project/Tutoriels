import numpy as np
import os
import cv2
import glob
import config

for image in glob.glob(config.dir_images+'*_HC.png'):
    image_ellipse=image.split('.')[0]+"_Annotation.png"
    img=cv2.imread(image_ellipse)
    img=cv2.resize(img, (config.largeur, config.hauteur))
    print(img.shape)
    h, w, c=img.shape
    img=img[:, :, 0]
    contours, hierarchy=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        (x, y),(ma, MA) ,angle = cv2.fitEllipse(cont)
        print("{}:{:f}:{:f}:{:f}:{:f}:{:f}".format(image, x/w, y/h, ma/w, MA/h, angle/180))
