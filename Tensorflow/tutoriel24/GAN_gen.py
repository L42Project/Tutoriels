from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import time
import math
import re

batch_size=128
epochs=10000
noise_dim=100
num_examples_to_generate=36

def make_generator_model():
    model=tf.keras.Sequential()

    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((4, 4, 1024)))

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))

    return model

generator=make_generator_model()
checkpoint=tf.train.Checkpoint(generator=generator)

checkpoint.restore("./training_checkpoints/ckpt-90")

d=50
flag=1
while True:
    if flag:
        bruit=tf.random.normal([1, noise_dim])
    test_images=generator(bruit, training=False)
    v=bruit[0][d]
    img=np.float32(test_images[0])
    print(">>> {}:{:6.8f}".format(d, v))
    cv2.imshow("image", img)
    flag=1
    key=cv2.waitKey()
    if key==ord('q'):
        quit()
    if key==ord('o'):
        d=min(100, d+1)
        flag=0
    if key==ord('l'):
        d=max(1, d-1)
        flag=0
    if key==ord('p'):
        u=np.zeros([1, noise_dim], dtype=np.float32)
        u[0][d]=0.5
        tu=tf.convert_to_tensor(u, dtype=tf.float32)
        bruit=tf.add(bruit, tu)
        flag=0
    if key==ord('m'):
        u=np.zeros([1, noise_dim], dtype=np.float32)
        u[0][d]=-0.5
        tu=tf.convert_to_tensor(u, dtype=tf.float32)
        bruit=tf.add(bruit, tu)
        flag=0
    
