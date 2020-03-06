import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math
import common
import config
import model

images, labels, labels2=common.read_json('test.json', 10)
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)

model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("./training/"))

dataset=tf.data.Dataset.from_tensor_slices((images, labels)).batch(config.batch_size)

def calcul_map(model, dataset, beta=1., seuil=0.5):
  grid=np.meshgrid(np.arange(config.cellule_x, dtype=np.float32), np.arange(config.cellule_y, dtype=np.float32))
  grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
  grid=np.tile(grid, (1, 1, 1, config.nbr_boxes, 1))

  index_labels2=0
  labels2_=labels2*[config.r_x, config.r_y, config.r_x, config.r_y, 1, 1, 1]
  score=[]
  tab_nbr_reponse=[]
  tab_tp=[]
  tab_true_boxes=[]
  
  for images, labels in dataset:
    predictions=np.array(model(images))

    pred_conf=common.sigmoid(predictions[:, :, :, :, 4])
    pred_classes=common.softmax(predictions[:, :, :, :, 5:])
    pred_ids=np.argmax(pred_classes, axis=-1)
    
    x_center=((grid[:, :, :, :, 0]+common.sigmoid(predictions[:, :, :, :, 0]))*config.r_x)
    y_center=((grid[:, :, :, :, 1]+common.sigmoid(predictions[:, :, :, :, 1]))*config.r_y)
    w=(np.exp(predictions[:, :, :, :, 2])*config.anchors[:, 0]*config.r_x)
    h=(np.exp(predictions[:, :, :, :, 3])*config.anchors[:, 1]*config.r_y)

    x_min=x_center-w/2
    y_min=y_center-h/2
    x_max=x_center+w/2
    y_max=y_center+h/2

    tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).astype(np.float32)
    tab_boxes=tab_boxes.reshape(-1, config.cellule_y*config.cellule_x*config.nbr_boxes, 4)
    pred_conf=pred_conf.reshape(-1, config.cellule_y*config.cellule_x*config.nbr_boxes)
    pred_ids=pred_ids.reshape(-1, config.cellule_y*config.cellule_x*config.nbr_boxes)

    for p in range(len(predictions)):
      nbr_reponse=np.zeros(config.nbr_classes)
      tp=np.zeros(config.nbr_classes)
      nbr_true_boxes=np.zeros(config.nbr_classes)
      tab_index=tf.image.non_max_suppression(tab_boxes[p], pred_conf[p], 100)      
      for id in tab_index:
        if pred_conf[p, id]>0.10:
          nbr_reponse[pred_ids[p, id]]+=1
          for box in labels2_[index_labels2]:
            if not box[5]:
              break
            b1=[tab_boxes[p, id, 1], tab_boxes[p, id, 0], tab_boxes[p, id, 3], tab_boxes[p, id, 2]]
            iou=common.intersection_over_union(b1, box)
            if iou>seuil and box[6]==pred_ids[p, id]:
              tp[pred_ids[p, id]]+=1

      for box in labels2[index_labels2]:
        if not box[5]:
          break
        nbr_true_boxes[int(box[6])]+=1

      tab_nbr_reponse.append(nbr_reponse)
      tab_tp.append(tp)
      tab_true_boxes.append(nbr_true_boxes)
      
      index_labels2=index_labels2+1

  tab_nbr_reponse=np.array(tab_nbr_reponse)
  tab_tp=np.array(tab_tp)
  tab_true_boxes=np.array(tab_true_boxes)

  ########################
  precision_globule_rouge=tab_tp[:, 1]/(tab_nbr_reponse[:, 1]+1E-7)
  precision_trophozoite=tab_tp[:, 4]/(tab_nbr_reponse[:, 4]+1E-7)

  rappel_globule_rouge=tab_tp[:, 1]/(tab_true_boxes[:, 1]+1E-7)
  rappel_trophozoite=tab_tp[:, 4]/(tab_true_boxes[:, 4]+1E-7)
    
  print("F1 score globule rouge", np.mean(2*precision_globule_rouge*rappel_globule_rouge/(precision_globule_rouge+rappel_globule_rouge+1E-7)))
  print("F1 score trophozoite", np.mean(2*precision_trophozoite*rappel_trophozoite/(precision_trophozoite+rappel_trophozoite+1E-7)))
  
  precision=(precision_globule_rouge+precision_trophozoite)/2
  rappel=(rappel_globule_rouge+rappel_trophozoite)/2

  score=np.mean((1+beta*beta)*precision*rappel/(beta*beta*precision+rappel+1E-7))
  print("SCORE (globule rouge/trophozoite)", score)
  ########################

  precision=tab_tp/(tab_nbr_reponse+1E-7)
  rappel=tab_tp/(tab_true_boxes+1E-7)
  score=np.mean((1+beta*beta)*precision*rappel/(beta*beta*precision+rappel+1E-7))
  
  return score

score=calcul_map(model, dataset)
print("Resultat", score)
