import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import time
import common
import dataset

batch_size=128
nbr_entrainement=20

tab_images=np.array([]).reshape(0, common.size, common.size, 3)
tab_labels=[]

tab_panneau, tab_image_panneau=common.lire_images_panneaux(common.dir_images_panneaux, common.size)

id=0
for image in tab_image_panneau:
    lot=dataset.create_lot_img(image, 12000)
    tab_images=np.concatenate((tab_images, lot))
    tab_labels=np.concatenate([tab_labels, np.full(len(lot), id)])
    id+=1
                
tab_panneau=np.array(tab_panneau)
tab_images=np.array(tab_images, dtype=np.float32)/255
tab_labels=np.array(tab_labels, dtype=np.float32).reshape([-1, 1])

train_images, test_images, train_labels, test_labels=train_test_split(tab_images, tab_labels, test_size=0.10)

train_ds=tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

print("train_images", len(train_images))
print("test_images", len(test_images))

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions=model_panneau(images)
    loss=loss_object(labels, predictions)
  gradients=tape.gradient(loss, model_panneau.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model_panneau.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

def train(train_ds, nbr_entrainement):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images, labels in train_ds:
      train_step(images, labels)
    message='Entrainement {:04d}: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
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
    predictions=model_panneau(test_images)
    t_loss=loss_object(test_labels, predictions)
    test_loss(t_loss)
    test_accuracy(test_labels, predictions)
  message='   >>> Test: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))
  test_loss.reset_states()
  test_accuracy.reset_states()

optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
model_panneau=common.panneau_model(len(tab_panneau))
checkpoint=tf.train.Checkpoint(model_panneau=model_panneau)

print("Entrainement")
train(train_ds, nbr_entrainement)
checkpoint.save(file_prefix="./training_panneau/panneau")

quit()
for i in range(len(test_images)):
    prediction=model_panneau([test_images[i]])
    print("prediction", prediction, tab_panneau[np.argmax(prediction[0])])
    cv2.imshow("image", test_images[i])
    cv2.waitKey()
