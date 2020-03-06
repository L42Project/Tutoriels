import numpy as np

dict={'leukocyte':     (255, 255, 0  ),
      'red blood cell':(0  , 0  , 255),
      'ring':          (0  , 255, 0  ),
      'schizont':      (255, 0  , 255),
      'trophozoite':   (255, 0  , 0  ),
      'difficult':     (0  , 0  , 0  ),
      'gametocyte':    (0  , 255, 255)}
dict2=[]
for d in dict:
    dict2.append(d)

largeur=256
hauteur=192
cellule_x=16
cellule_y=12
nbr_classes=len(dict)
r_x=int(largeur/cellule_x)
r_y=int(hauteur/cellule_y)
max_objet=60

anchors=np.array([[3.0, 1.5], [2.0, 2.0], [1.5, 3.0]])
nbr_boxes=len(anchors)

batch_size=16

lambda_coord=5
lambda_noobj=0.5
#lambda_coord=1
#lambda_noobj=1
    
seuil_iou_loss=0.6
