import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2

size=42
dir_images_panneaux="images_panneaux"
dir_images_autres_panneaux="images_autres_panneaux"
dir_images_sans_panneaux="images_sans_panneaux"

def lire_images_panneaux(dir_images_panneaux, size=None):
    tab_panneau=[]
    tab_image_panneau=[]

    if not os.path.exists(dir_images_panneaux):
        quit("Le repertoire d'image n'existe pas: {}".format(dir_images_panneaux))

    files=os.listdir(dir_images_panneaux)
    if files is None:
        quit("Le repertoire d'image est vide: {}".format(dir_images_panneaux))

    for file in sorted(files):
        if file.endswith("png"):
            tab_panneau.append(file.split(".")[0])
            image=cv2.imread(dir_images_panneaux+"/"+file)
            if size is not None:
                image=cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
            tab_image_panneau.append(image)
            
    return tab_panneau, tab_image_panneau
