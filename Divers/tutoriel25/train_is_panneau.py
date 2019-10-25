import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import common
import time
import dataset

batch_size=64
nbr_entrainement=20

tab_images=np.array([]).reshape(0, common.size, common.size, 3)

tab_panneau, tab_image_panneau=common.lire_images_panneaux(common.dir_images_panneaux, common.size)

if not os.path.exists(common.dir_images_autres_panneaux):
    quit("Le repertoire d'image n'existe pas: {}".format(common.dir_images_autres_panneaux))

if not os.path.exists(common.dir_images_sans_panneaux):
    quit("Le repertoire d'image n'existe pas:".format(common.dir_images_sans_panneaux))

nbr=0
for image in tab_image_panneau:
    lot=dataset.create_lot_img(image, 12000)
    tab_images=np.concatenate([tab_images, lot])
    nbr+=len(lot)

tab_labels=np.full(nbr, 1)

print("Image panneaux:", nbr)

files=os.listdir(common.dir_images_autres_panneaux)
if files is None:
    quit("Le repertoire d'image est vide:".format(common.dir_images_autres_panneaux))

nbr=0
for file in files:
    if file.endswith("png"):
        path=os.path.join(common.dir_images_autres_panneaux, file)
        image=cv2.resize(cv2.imread(path), (common.size, common.size), cv2.INTER_LANCZOS4)
        lot=dataset.create_lot_img(image, 700)
        tab_images=np.concatenate([tab_images, lot])
        nbr+=len(lot)

tab_labels=np.concatenate([tab_labels, np.full(nbr, 0)])
        
print("Image autres panneaux:", nbr)

nbr_np=int(len(tab_images)/2)
print("nbr_np", nbr_np)

id=1
nbr=0
tab=[]
for cpt in range(nbr_np):
    file=common.dir_images_sans_panneaux+"/{:d}.png".format(id)
    if not os.path.isfile(file):
        break
    image=cv2.resize(cv2.imread(file), (common.size, common.size))
    #tab_images.append(image)
    tab.append(image)
    id+=1
    nbr+=1

tab_images=np.concatenate([tab_images, tab])
tab_labels=np.concatenate([tab_labels, np.full(nbr, 0)])
print("Image sans panneaux:", nbr)

tab_images=np.array(tab_images, dtype=np.float32)/255
tab_labels=np.array(tab_labels, dtype=np.float32).reshape([-1, 1])

tab_images, tab_labels=shuffle(tab_images, tab_labels)
train_images, test_images, train_labels, test_labels=train_test_split(tab_images, tab_labels, test_size=0.10)

train_ds=tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

print("train_images", len(train_images))
print("test_images", len(test_images))
print("nbr panneau", len(np.where(train_labels==0.)[1]), train_labels.shape)

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions=model_is_panneau(images)
    loss=loss_object(labels, predictions)
  gradients=tape.gradient(loss, model_is_panneau.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model_is_panneau.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

def train(train_ds, nbr_entrainement):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images, labels in train_ds:
      train_step(images, labels)
    message='Entrainement {:04d}, loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(entrainement+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         time.time()-start))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test(test_ds)

def test(test_ds):
  start=time.time()
  for test_images, test_labels in test_ds:
    predictions=model_is_panneau(test_images)
    t_loss=loss_object(test_labels, predictions)
    test_loss(t_loss)
    test_accuracy(test_labels, predictions)
  message='   >>> Test: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))

optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.BinaryCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.BinaryAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.BinaryAccuracy()
model_is_panneau=common.is_panneau_model()
checkpoint=tf.train.Checkpoint(model_is_panneau=model_is_panneau)

print("Entrainement")
train(train_ds, nbr_entrainement)
test(test_ds)

checkpoint.save(file_prefix="./training_is_panneau/is_panneau")
