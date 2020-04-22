import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import common
import config
import model

images, labels=common.prepare_data('training_set.csv')
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)
index=np.random.permutation(len(images))
images=images[index].reshape(-1, config.hauteur, config.largeur, 1)
labels=labels[index]

print("Nbr images:", len(images))

train_ds=tf.data.Dataset.from_tensor_slices((images, labels)).batch(config.batch_size)

del images
del labels

def my_loss(labels, preds):
    lambda_xy=5
    lambda_Aa=5
    lambda_angle=1
    
    preds_xy=preds[:, 0:2]
    preds_Aa=preds[:, 2:4]
    preds_angle=preds[:, 4]

    labels_xy=labels[:, 0:2]
    labels_Aa=labels[:, 2:4]
    labels_angle=labels[:, 4]
    
    loss_xy=tf.reduce_sum(tf.math.square(preds_xy-labels_xy), axis=-1)
    loss_Aa=tf.reduce_sum(tf.math.square(preds_Aa-labels_Aa), axis=-1)
    loss_angle=tf.math.square(preds_angle-labels_angle)
    
    loss=lambda_xy*loss_xy+lambda_Aa*loss_Aa+lambda_angle*loss_angle
    return loss

model=model.model(config.input_model)

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=my_loss(labels, predictions)
  gradients=tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)

def train(train_ds, nbr_entrainement):
    for entrainement in range(nbr_entrainement):
        start=time.time()
        for images, labels in train_ds:
            train_step(images, labels)
        message='Entrainement {:04d}: loss: {:6.4f}, temps: {:7.4f}'
        print(message.format(entrainement+1,
                             train_loss.result(),
                             time.time()-start))
        if not entrainement%10:
            checkpoint.save(file_prefix="./training/")
    
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
checkpoint=tf.train.Checkpoint(model=model)
train_loss=tf.keras.metrics.Mean()

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("./training/"))

train(train_ds, 60)
checkpoint.save(file_prefix="./training/")
