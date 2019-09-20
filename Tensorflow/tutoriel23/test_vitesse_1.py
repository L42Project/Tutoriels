import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test)=mnist.load_data()
batch_size=64
epochs=5

(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_train=(x_train.reshape(-1, 28, 28, 1)/255).astype(np.float32)
x_test=(x_test.reshape(-1, 28, 28, 1)/255).astype(np.float32)

train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model = tf.keras.models.Sequential([
    layers.Conv2D(64,  3, strides=2, activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128,  3, strides=2, activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs)
#model.evaluate(x_test, y_test)
