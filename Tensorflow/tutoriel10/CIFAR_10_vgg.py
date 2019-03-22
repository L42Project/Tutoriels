import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import cv2
import vgg
from sklearn.utils import shuffle

def read_cifar_file(file, images, labels):
    shift=0
    f=np.fromfile(file, dtype=np.uint8)
    while shift!=f.shape[0]:
        labels.append(np.eye(10)[f[shift]])
        shift+=1
        images.append(f[shift:shift+3*32*32].reshape(3, 32, 32).transpose(1, 2, 0)/255)
        shift+=3*32*32

taille_batch=100
nbr_entrainement=50
labels=['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

train_images=[]
train_labels=[]
read_cifar_file("cifar-10-batches-bin/data_batch_1.bin", train_images, train_labels)
read_cifar_file("cifar-10-batches-bin/data_batch_2.bin", train_images, train_labels)
read_cifar_file("cifar-10-batches-bin/data_batch_3.bin", train_images, train_labels)
read_cifar_file("cifar-10-batches-bin/data_batch_4.bin", train_images, train_labels)
read_cifar_file("cifar-10-batches-bin/data_batch_5.bin", train_images, train_labels)

test_images=[]
test_labels=[]
read_cifar_file("cifar-10-batches-bin/test_batch.bin", test_images, test_labels)
    
images, labels, is_training, sortie, train, accuracy, save=vgg.vggnet(nbr_classes=10, learning_rate=0.01)

def transform_img(img):
    img=tf.image.random_flip_left_right(img)
    img=tf.image.random_hue(img, 0.08)
    img=tf.image.random_saturation(img, 0.6, 1.6)
    img=tf.image.random_brightness(img, 0.05)
    img=tf.image.random_contrast(img, 0.7, 1.3)
    x=int(img.shape[0])
    y=int(img.shape[1])
    z=int(img.shape[2])
    img=tf.image.random_crop(img, [int(x*0.90), int(y*0.90), z])
    img=tf.image.resize_images(img, (x, y))
    return(img)

fichier=open("log", "a")
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]

    train_images=np.array(train_images, dtype=np.float32)
    train_images2=tf.map_fn(transform_img, train_images)
    train_images3=tf.map_fn(transform_img, train_images)
    train_images4=tf.map_fn(transform_img, train_images)
    train_images=tf.concat([train_images, train_images2, train_images3, train_images4], axis=0)
    train_labels=np.array(train_labels)
    train_labels=tf.concat([train_labels, train_labels, train_labels, train_labels], axis=0)

    train_images=s.run(train_images)
    train_labels=s.run(train_labels)
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
     

