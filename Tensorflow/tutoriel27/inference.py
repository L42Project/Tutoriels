import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math
import common
import config
import model

images, labels, labels2=common.read_json('test.json', 5)
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)
index=np.random.permutation(len(images))
images=images[index]
labels=labels[index]

model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("./training/"))

grid=np.meshgrid(np.arange(config.cellule_x, dtype=np.float32), np.arange(config.cellule_y, dtype=np.float32))
grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
grid=np.tile(grid, (1, 1, config.nbr_boxes, 1))

for i in range(len(images)):
  img=common.prepare_image(images[i], labels[i], False)
  predictions=model(np.array([images[i]]))
  
  pred_boxes=predictions[0, :, :, :, 0:4]
  pred_conf=common.sigmoid(predictions[0, :, :, :, 4])
  pred_classes=common.softmax(predictions[0, :, :, :, 5:])
  ids=np.argmax(pred_classes, axis=-1)

  x_center=((grid[:, :, :, 0]+common.sigmoid(pred_boxes[:, :, :, 0]))*config.r_x)
  y_center=((grid[:, :, :, 1]+common.sigmoid(pred_boxes[:, :, :, 1]))*config.r_y)
  w=(np.exp(pred_boxes[:, :, :, 2])*config.anchors[:, 0]*config.r_x)
  h=(np.exp(pred_boxes[:, :, :, 3])*config.anchors[:, 1]*config.r_y)

  x_min=(x_center-w/2).astype(np.int32)
  y_min=(y_center-h/2).astype(np.int32)
  x_max=(x_center+w/2).astype(np.int32)
  y_max=(y_center+h/2).astype(np.int32)
  
  tab_boxes=[]
  conf=[]
  for y in range(config.cellule_y):
    for x in range(config.cellule_x):
      for b in range(config.nbr_boxes):
        if pred_conf[y, x, b]>0.10:
          color=list(config.dict.values())[ids[y, x, b]]
          cv2.circle(images[i], (x_center[y, x, b], y_center[y, x, b]), 1, color, 2)
          cv2.rectangle(images[i], (x_min[y, x, b], y_min[y, x, b]), (x_max[y, x, b], y_max[y, x, b]), color, 1)
          cv2.rectangle(images[i], (x_min[y, x, b], y_min[y, x, b]), (x_max[y, x, b], y_min[y, x, b]-15), color, cv2.FILLED)
          cv2.putText(images[i], "{:3.0%}".format(pred_conf[y, x, b]), (x_min[y, x, b], y_min[y, x, b]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (255, 255, 255), 1)

  cv2.imshow("Inference", images[i])
  cv2.imshow("Bonne reponse", img)
    
  key=cv2.waitKey()&0xFF
  if key==ord('q'):
    quit()
