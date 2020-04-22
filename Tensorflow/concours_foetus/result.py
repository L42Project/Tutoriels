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

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("./training/"))

print("filename,center_x_mm,center_y_mm,semi_axes_a_mm,semi_axes_b_mm,angle_rad")
with open("test_set.csv", newline='') as csvfile:
  lignes=csv.reader(csvfile, delimiter=',')
  for ligne in lignes:
    img=cv2.imread(config.dir_images_test+ligne[0])
    if img is None:
      continue
    mm_pixel=float(ligne[1])
    H, W, C=img.shape
    img=cv2.resize(img, (config.largeur, config.hauteur))
    img=np.array(img, dtype=np.float32)/255
    img=np.expand_dims(img[:, :, 0], axis=-1)
    predictions=model(np.array([img]))
    x, y, grand_axe, petit_axe, angle=predictions[0]

    x=float(x*W*mm_pixel)
    y=float(y*W*mm_pixel)
    axis_x=float(grand_axe*W*mm_pixel/2)
    axis_y=float(petit_axe*W*mm_pixel/2)
    
    r=np.pi
    r_2=r/2
    if angle>=0.5:
      angle=angle*r-r_2
    else:
      angle=angle*r+r_2

    print("{},{:f},{:f},{:f},{:f},{:f}".format(ligne[0], x, y, axis_x, axis_y, angle))

