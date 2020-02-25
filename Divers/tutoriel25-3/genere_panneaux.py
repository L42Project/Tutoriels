import numpy as np
from sklearn.utils import shuffle
import cv2
import common
import dataset

tab_panneau, tab_image_panneau=common.lire_images_panneaux(common.dir_images_panneaux, common.size)

tab_images=np.array([]).reshape(0, common.size, common.size, 3)
tab_labels=[]

id=0
for image in tab_image_panneau:
    lot=dataset.create_lot_img(image, 1000)
    tab_images=np.concatenate([tab_images, lot])
    tab_labels=np.concatenate([tab_labels, np.full(len(lot), id)])
    id+=1

tab_panneau=np.array(tab_panneau)
tab_images=np.array(tab_images, dtype=np.float32)/255
tab_labels=np.array(tab_labels).reshape([-1, 1])

tab_images, tab_labels=shuffle(tab_images, tab_labels)

for i in range(len(tab_images)):
    cv2.imshow("panneau", tab_images[i])
    print("label", tab_labels[i], "panneau", tab_panneau[int(tab_labels[i])])
    if cv2.waitKey()&0xFF==ord('q'):
        quit()
