import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2

size=42
dir_images_panneaux="images_panneaux"
dir_images_autres_panneaux="images_autres_panneaux"
dir_images_sans_panneaux="images_sans_panneaux"

def panneau_model(nbr_classes):
    model=tf.keras.Sequential()

    model.add(layers.Input(shape=(size, size, 3), dtype='float32'))
    
    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(nbr_classes, activation='sigmoid'))
    
    return model

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
