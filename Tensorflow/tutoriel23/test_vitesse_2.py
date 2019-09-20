import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle

batch_size=16
epochs=5

def stl10(path):
    labels=['avion', 'oiseau', 'voiture', 'chat', 'cerf', 'chien', 'cheval', 'singe', 'bateau', 'camion']
    train_images=np.fromfile(path+"/train_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)
    train_labels=np.fromfile(path+"/train_y.bin", dtype=np.uint8)-1
    train_images, train_labels=shuffle(train_images, train_labels)
    test_images=np.fromfile(path+"/test_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)
    test_labels=np.fromfile(path+"/test_y.bin", dtype=np.uint8)-1
    return labels, train_images, train_labels, test_images, test_labels

labels, x_train, y_train, x_test, y_test=stl10("stl10_binary")
x_train=(x_train/255).astype(np.float32)
x_test=(x_test/255).astype(np.float32)

train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model=models.Sequential([
    layers.Conv2D(256, 5, strides=1),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Conv2D(512, 5, strides=1),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Conv2D(1024, 5, strides=1),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Conv2D(2048, 5, strides=1),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs)
#model.evaluate(x_test, y_test)
#model.summary()
