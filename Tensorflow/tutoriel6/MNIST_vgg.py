import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import vgg

taille_batch=100
nbr_entrainement=200

train_images=np.fromfile("mnist/train-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
train_labels=np.eye(10)[np.fromfile("mnist/train-labels-idx1-ubyte", dtype=np.uint8)[8:]]
test_images=np.fromfile("mnist/t10k-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
test_labels=np.eye(10)[np.fromfile("mnist/t10k-labels-idx1-ubyte", dtype=np.uint8)[8:]]

images, labels, is_training, sortie, train, accuracy, saver=vgg.vggnet()

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]
    train_images, train_labels=shuffle(train_images, train_labels)
    for id_entrainement in np.arange(nbr_entrainement):
        print("> Entrainement", id_entrainement)
        for batch in np.arange(0, len(train_images), taille_batch):
            s.run(train, feed_dict={
                images: train_images[batch:batch+taille_batch],
                labels: train_labels[batch:batch+taille_batch],
                is_training: True
            })
        print("  entrainement OK")
        tab_accuracy_train=[]
        for batch in np.arange(0, len(train_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                images: train_images[batch:batch+taille_batch],
                labels: train_labels[batch:batch+taille_batch],
                is_training: True
            })
            tab_accuracy_train.append(p)
        print("  train:", np.mean(tab_accuracy_train))
        tab_accuracy_test=[]
        for batch in np.arange(0, len(test_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                images: test_images[batch:batch+taille_batch],
                labels: test_labels[batch:batch+taille_batch],
                is_training: True
            })
            tab_accuracy_test.append(p)
        print("  test :", np.mean(tab_accuracy_test))
        tab_train.append(1-np.mean(tab_accuracy_train))
        tab_test.append(1-np.mean(tab_accuracy_test))
    saver.save(s, './mon_vgg/modele')
