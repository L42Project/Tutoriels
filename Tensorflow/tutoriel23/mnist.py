import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time

batch_size=64
nbr_entrainement=5

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train=(x_train.reshape(-1, 28, 28, 1)/255).astype(np.float32)
x_test=(x_test.reshape(-1, 28, 28, 1)/255).astype(np.float32)

train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model = models.Sequential([
    layers.Conv2D(64,  3, strides=2, activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128,  3, strides=2, activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=loss_object(labels, predictions)
  gradients=tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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

def test(test_ds):
  start=time.time()
  for test_images, test_labels in test_ds:
    predictions=model(test_images)
    t_loss=loss_object(test_labels, predictions)
    test_loss(t_loss)
    test_accuracy(test_labels, predictions)
  message='Loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))

print("Entrainement")
train(train_ds, nbr_entrainement)

print("Jeu de test")
test(test_ds)

