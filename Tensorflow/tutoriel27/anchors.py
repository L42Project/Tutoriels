import json
import cv2
import numpy as np
import common
import config

def create_anchors():
  tab_anchors=[]
  mid_x_cell=int(config.r_x/2)
  mid_y_cell=int(config.r_y/2)
  for y in range(mid_y_cell, config.hauteur, config.r_y):
    for x in range(mid_x_cell, config.largeur, config.r_x):
      t_anchor=[]
      for a in range(len(config.anchors)):
        dx=int((config.anchors[a][0]*config.r_x)/2)
        dy=int((config.anchors[a][1]*config.r_y)/2)
        t_anchor.append([max(0, x-dx), max(0, y-dy), min(x+dx, config.largeur-1), min(y+dy, config.hauteur-1)])
      tab_anchors.append(t_anchor)
  tab_anchors=np.array(tab_anchors)
  return tab_anchors

images, labels, labels2=common.read_json('training.json', 5, 20)

cell_x=0
cell_y=0
i=0
tab_anchors=create_anchors()

for i in range(len(images)):
    while True:
        img=common.prepare_image(images[i], labels[i])
        for a in tab_anchors[cell_y*config.cellule_x+cell_x]:
            cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (255, 255, 0), 1)
        print(labels[i, cell_y, cell_x])
        cv2.imshow("image", cv2.resize(img, (2*config.largeur, 2*config.hauteur)))
        key=cv2.waitKey()&0xFF
        if key==ord('f'):
            cell_x=min(config.cellule_x-1, cell_x+1)
        if key==ord('s'):
            cell_x=max(0, cell_x-1)
        if key==ord('d'):
            cell_y=min(config.cellule_y-1, cell_y+1)
        if key==ord('e'):
            cell_y=max(0, cell_y-1)
        if key==ord('n'):
            break
        if key==ord('q'):
            quit()

