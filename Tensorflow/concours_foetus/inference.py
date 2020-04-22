import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math
import common
import config
import model
import csv

model=model.model(config.input_model)

rouge=(0, 0, 255)
vert=(0, 255, 0)

if True:
  dir=config.dir_images
  fichier="training_set.csv"
  test=False
else:
  dir=config.dir_images_test
  fichier="test_set.csv"
  test=True
  
checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("./training/"))

with open(fichier, newline='') as csvfile:
  lignes=csv.reader(csvfile, delimiter=',')
  for ligne in lignes:
    print("LIGNE:", ligne)
    print("XXX", dir+ligne[0])
    img_originale=cv2.imread(dir+ligne[0])
    if img_originale is None:
      continue
    print("WWW", ligne[0], dir+ligne[0], img_originale.shape)
    H, W, C=img_originale.shape
    mm_pixel=float(ligne[1])
    img=cv2.resize(img_originale, (config.largeur, config.hauteur))
    img2=img.copy()
    img=np.array(img, dtype=np.float32)/255
    img=np.expand_dims(img[:, :, 0], axis=-1)
    predictions=model(np.array([img]))
    x, y, grand_axe, petit_axe, angle=predictions[0]
    cv2.ellipse(img2, (x*config.norm, y*config.norm), (petit_axe*config.norm/2, grand_axe*config.norm/2), angle*180, 0., 360., rouge, 2)
    print("Prediction", np.array(predictions[0]))

    if test is False:
      f_ellipse=ligne[0].split('.')[0]+"_Annotation.png"
      image_ellipse=cv2.imread(dir+f_ellipse)
      if image_ellipse is None:
        print("Fichier absent", dir+f_ellipse)
        continue
      img_ellipse_f_=image_ellipse[:, :, 0]
      contours, hierarchy=cv2.findContours(img_ellipse_f_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      (x_, y_), (ma_, MA_), a_=cv2.fitEllipse(contours[0])
      cv2.ellipse(img_originale, (int(x_), int(y_)), (int(ma_/2), int(MA_/2)), a_, 0., 360., vert, 3)    

    cv2.ellipse(img_originale, (x*W, y*W), (petit_axe*W/2, grand_axe*W/2), angle*180, 0., 360., rouge, 2)

    x=float(x*W*mm_pixel)
    y=float(y*W*mm_pixel)
    axis_x=float(grand_axe*W*mm_pixel/2)
    axis_y=float(petit_axe*W*mm_pixel/2)
    
    r=180.
    r_2=r/2
    if angle>=0.5:
      angle=angle*r-r_2
    else:
      angle=angle*r+r_2

    HC=np.pi*np.sqrt(2*(axis_x**2+axis_y**2))
    
    cv2.putText(img_originale, "HC: {:5.2f}mm".format(HC), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    print("{},{:f},{:f},{:f},{:f},{:f}    HC: {:f}mm".format(ligne[0], x, y, axis_x, axis_y, angle, HC))
    if len(ligne)==3:
      cv2.putText(img_originale, "HC: {:5.2f}mm".format(float(ligne[2])), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
      print("HC: {}mm   prediction: {:f}mm".format(ligne[2], HC))
    
    cv2.imshow("Image originale", img_originale)
    cv2.imshow("Inference", img2)
    
    key=cv2.waitKey()&0xFF
    if key==ord('q'):
      quit()
