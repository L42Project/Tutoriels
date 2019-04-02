import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plot
import cv2

def read_cifar_file(file, images, labels):
    shift=0
    f=np.fromfile(file, dtype=np.uint8)
    while shift!=f.shape[0]:
        labels.append(np.eye(10)[f[shift]])
        shift+=1
        images.append(f[shift:shift+3*32*32].reshape(3, 32, 32).transpose(1, 2, 0)/255)
        shift+=3*32*32

def convolution(couche_prec, taille_noyau, nbr_noyau):
    w=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_prec.get_shape()[-1]), nbr_noyau)))
    b=np.zeros(nbr_noyau)
    result=tf.nn.conv2d(couche_prec, w, strides=[1, 1, 1, 1], padding='SAME')+b
    return result
        
def fc(couche_prec, nbr_neurone):
    w=tf.Variable(tf.random.truncated_normal(shape=(int(couche_prec.get_shape()[-1]), nbr_neurone), dtype=tf.float32))
    b=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(couche_prec, w)+b
    return result

def normalisation(couche_prec):
    mean, var=tf.nn.moments(couche_prec, [0])
    scale=tf.Variable(tf.ones(shape=(np.shape(couche_prec)[-1])))
    beta=tf.Variable(tf.zeros(shape=(np.shape(couche_prec)[-1])))
    result=tf.nn.batch_normalization(couche_prec, mean, var, beta, scale, 0.001)
    return result

taille_batch=100
nbr_entrainement=200
learning_rate=0.01
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
    
ph_images=tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)

result=convolution(ph_images, 3, 64)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 64)
result=normalisation(result)
result=tf.nn.relu(result)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result=convolution(result, 3, 128)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 128)
result=normalisation(result)
result=tf.nn.relu(result)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result=convolution(result, 3, 256)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 256)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 256)
result=normalisation(result)
result=tf.nn.relu(result)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result=convolution(result, 3, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result=convolution(result, 3, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=convolution(result, 3, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
result=tf.contrib.layers.flatten(result)

result=fc(result, 512)
result=normalisation(result)
result=tf.nn.relu(result)
result=fc(result, 10)
socs=tf.nn.softmax(result)

erreur=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=result)
train=tf.train.AdamOptimizer(learning_rate).minimize(erreur)
precision=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(socs, 1), tf.argmax(ph_labels, 1)), tf.float32))


with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]
    train_images, train_labels=shuffle(train_images, train_labels)
    for id_entrainement in np.arange(nbr_entrainement):
        print("> Entrainement", id_entrainement)
        for batch in np.arange(0, len(train_images), taille_batch):
            s.run(train, feed_dict={
                ph_images: train_images[batch:batch+taille_batch],
                ph_labels: train_labels[batch:batch+taille_batch],
            })
        print("  entrainement OK")
        tab_precision_train=[]
        for batch in np.arange(0, len(train_images), taille_batch):
            p=s.run(precision, feed_dict={
                ph_images: train_images[batch:batch+taille_batch],
                ph_labels: train_labels[batch:batch+taille_batch]
            })
            tab_precision_train.append(p)
        print("  train:", np.mean(tab_precision_train))
        tab_precision_test=[]
        for batch in np.arange(0, len(test_images), taille_batch):
            p=s.run(precision, feed_dict={
                ph_images: test_images[batch:batch+taille_batch],
                ph_labels: test_labels[batch:batch+taille_batch]
            })
            tab_precision_test.append(p)
        print("  test :", np.mean(tab_precision_test))
        tab_train.append(1-np.mean(tab_precision_train))
        tab_test.append(1-np.mean(tab_precision_test))
    quit()
    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_train, label="Train error")
    plot.plot(tab_test, label="Test error")
    plot.legend(loc="upper right")
    plot.show()
    
    resulat=s.run(socs, feed_dict={ph_images: test_images[0:taille_batch]})
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    for image in range(taille_batch):
        print("image", image)
        print("sortie du réseau:", resulat[image], np.argmax(resulat[image]), labels[np.argmax(resulat[image])])
        print("sortie attendue :", test_labels[image], np.argmax(test_labels[image]), labels[np.argmax(test_labels[image])])
        cv2.imshow('image', test_images[image])
        if cv2.waitKey()&0xFF==ord('q'):
            break

