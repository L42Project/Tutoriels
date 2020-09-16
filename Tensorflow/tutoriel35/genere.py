import tensorflow as tf
from tensorflow.keras import layers, models
import time, threading
import numpy as np
import cv2
import model_cond

noise_dim=100

generator=model_cond.generator_model()
checkpoint=tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir='./training_checkpoints_gan_cond/'))

marge=20

while True:
    chiffres=input("Entrez une serie de chiffre:")
    try:
        chiffres_int=int(chiffres)
    except:
        continue

    liste_chiffres=[]
    while (chiffres_int):
        liste_chiffres.append(chiffres_int%10)
        chiffres_int=int(chiffres_int/10)
    seed=tf.random.normal([len(liste_chiffres), noise_dim])
    labels=tf.one_hot(liste_chiffres, 10)
    image=np.zeros(shape=(28+2*marge, len(liste_chiffres)*28+2*marge), dtype=np.float32)
    prediction=generator([seed, labels], training=False)
    for i in range(len(prediction)):
        image[marge:marge+28, marge+i*28:marge+(i+1)*28]=prediction[len(liste_chiffres)-i-1, :, :, 0]*127.5+127.5
    cv2.imshow("Image", image.astype(np.uint8))
    key=cv2.waitKey(10)
