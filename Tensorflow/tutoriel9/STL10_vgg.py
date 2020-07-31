import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plot
import cv2
import vgg

labels=['avion', 'oiseau', 'voiture', 'chat', 'cerf', 'chien', 'cheval', 'singe', 'bateau', 'camion']
train_images=np.fromfile("stl10_binary/train_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)/255
train_labels=np.eye(10)[np.fromfile("stl10_binary/train_y.bin", dtype=np.uint8)-1]
test_images=np.fromfile("stl10_binary/test_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)/255
test_labels=np.eye(10)[np.fromfile("stl10_binary/test_y.bin", dtype=np.uint8)-1]

taille_batch=100
nbr_entrainement=200

images, labels, is_training, sortie, train, accuracy, save=vgg.vggnet(nbr_classes=10, learning_rate=0.001)

#train_images=tf.image.resize_images(train_images, size=[32, 32])
#test_images=tf.image.resize_images(train_images, size=[32, 32])

fichier=open("log", "a")
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
                is_training: False
            })
            tab_accuracy_train.append(p)
        print("  train:", np.mean(tab_accuracy_train))
        tab_accuracy_test=[]
        for batch in np.arange(0, len(test_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                images: test_images[batch:batch+taille_batch],
                labels: test_labels[batch:batch+taille_batch],
                is_training: False
            })
            tab_accuracy_test.append(p)
        print("  test :", np.mean(tab_accuracy_test))
        tab_train.append(1-np.mean(tab_accuracy_train))
        tab_test.append(1-np.mean(tab_accuracy_test))
        fichier.write("{:d}:{:f}:{:f}\n".format(id_entrainement, np.mean(tab_accuracy_train), np.mean(tab_accuracy_test)))
    fichier.close()
     

