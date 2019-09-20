from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import time
import math
import re

taille_batch=128
nbr_entrainement=10000
bruit_dim=100
nbr_exemples=36
dir_faces='faces/'

def generateur_model():
    model=tf.keras.Sequential()

    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(bruit_dim,)))
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

def discriminateur_model():    
    model=tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation=tf.nn.relu))
    model.add(layers.Dense(1))

    return model

@tf.function
def train_step(vrais_visages):
    bruit=tf.random.normal([taille_batch, bruit_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        faux_visages=generateur(bruit, training=True)

        prediction_vrais_visages=discriminateur(vrais_visages, training=True)
        prediction_faux_visages =discriminateur(faux_visages , training=True)

        generateur_loss    =cross_entropy(tf.ones_like (prediction_faux_visages ), prediction_faux_visages)
        discriminateur_loss=cross_entropy(tf.ones_like (prediction_vrais_visages), prediction_vrais_visages)+\
                            cross_entropy(tf.zeros_like(prediction_faux_visages ), prediction_faux_visages )     

    gradients_generateur    =gen_tape.gradient(generateur_loss     , generateur.trainable_variables)
    gradients_discriminateur=disc_tape.gradient(discriminateur_loss, discriminateur.trainable_variables)

    generateur_optimizer.apply_gradients    (zip(gradients_generateur    , generateur.trainable_variables))
    discriminateur_optimizer.apply_gradients(zip(gradients_discriminateur, discriminateur.trainable_variables))

def train(dataset, nbr_entrainement, bruit_pour_exemple=None):
    m=0
    for file in sorted(os.listdir('.')):
        f=re.search('img_(.+?).png', file)
        if f:
            m=int(f.group(1))
    checkpoint.restore(tf.train.latest_checkpoint("./training_checkpoints/"))
    for entrainement in range(m, nbr_entrainement):
        start=time.time()
        for image_batch in dataset:
            train_step(image_batch)
        if bruit_pour_exemple is not None:
            generatation_exemples(generateur, entrainement+1, bruit_pour_exemple)
        if (entrainement+1)%100==0:
            checkpoint.save(file_prefix="./training_checkpoints/ckpt")
        print ('Entrainement {}: temps {} secondes'.format(entrainement+1, time.time()-start))

def generatation_exemples(model, entrainement, bruit_pour_exemple):
    test_images=model(bruit_pour_exemple, training=False)
    n=int(math.sqrt(len(bruit_pour_exemple)))
    tab_img_test=np.zeros(shape=(128*n, 128*n, 3), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            tab_img_test[i*128:(i+1)*128, j*128:(j+1)*128, :]=test_images[i*n+j]
    cv2.imwrite('img_{:05d}.png'.format(entrainement), tab_img_test*255)
        
train_images=[]
for file in os.listdir(dir_faces):
    if file.endswith("jpg"):
        img=cv2.imread(dir_faces+file, cv2.IMREAD_COLOR)
        if img is not None:
            train_images.append(cv2.resize(img, (128, 128)))
train_images=np.array(train_images, dtype=np.float32)/255

generateur=generateur_model()
discriminateur=discriminateur_model()
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
generateur_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminateur_optimizer=tf.keras.optimizers.Adam(1e-4)
checkpoint=tf.train.Checkpoint(generateur_optimizer=generateur_optimizer,
                               discriminateur_optimizer=discriminateur_optimizer,
                               generateur=generateur,
                               discriminateur=discriminateur)
bruit_pour_exemple=tf.random.normal([nbr_exemples, bruit_dim])
train_dataset=tf.data.Dataset.from_tensor_slices(train_images).batch(taille_batch)
train(train_dataset ,nbr_entrainement, bruit_pour_exemple)
