import tensorflow as tf
import numpy as np
import random
from sklearn.utils import shuffle
import common

taille_batch=55
nbr_entrainement=400
learning_rate=1E-3

labels, train_images, train_labels, test_images, test_labels=common.stl10("stl10_binary")
train_images=train_images/255
test_images=test_images/255

ph_images, ph_labels, ph_is_training, socs, train, accuracy, saver=common.resnet(10, common.b_resnet_3M, learning_rate)

fichier=open("log", "a")
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]
    for id_entrainement in np.arange(nbr_entrainement):
        print("> Entrainement", id_entrainement)
        train_images, train_labels=shuffle(train_images, train_labels)
        for batch in np.arange(0, len(train_images), taille_batch):
            s.run(train, feed_dict={
                ph_images: train_images[batch:batch+taille_batch],
                ph_labels: train_labels[batch:batch+taille_batch],
                ph_is_training: True
            })
        print("  entrainement OK")
        tab_accuracy_train=[]
        for batch in np.arange(0, len(train_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                ph_images: train_images[batch:batch+taille_batch],
                ph_labels: train_labels[batch:batch+taille_batch],
                ph_is_training: True
            })
            tab_accuracy_train.append(p)
        print("  train:", np.mean(tab_accuracy_train))
        tab_accuracy_test=[]
        for batch in np.arange(0, len(test_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                ph_images: test_images[batch:batch+taille_batch],
                ph_labels: test_labels[batch:batch+taille_batch],
                ph_is_training: True
            })
            tab_accuracy_test.append(p)
        print("  test :", np.mean(tab_accuracy_test))
        tab_train.append(1-np.mean(tab_accuracy_train))
        tab_test.append(1-np.mean(tab_accuracy_test))
        fichier.write("{:d}:{:f}:{:f}\n".format(id_entrainement, np.mean(tab_accuracy_train), np.mean(tab_accuracy_test)))
    fichier.close()
