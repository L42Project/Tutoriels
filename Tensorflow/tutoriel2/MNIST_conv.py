import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import cv2

def convolution(couche_prec, taille_noyau, nbr_noyau):
    w_filtre=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_prec.get_shape()[-1]), nbr_noyau)))
    b_filtre=np.zeros(nbr_noyau)
    result=tf.nn.conv2d(couche_prec, w_filtre, strides=[1, 1, 1, 1], padding='SAME')+b_filtre
    return result
        
def fc(couche_prec, nbr_neurone):
    w=tf.Variable(tf.random.truncated_normal(shape=(int(couche_prec.get_shape()[-1]), nbr_neurone), dtype=tf.float32))
    b=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(couche_prec, w)+b
    return result

taille_batch=50
nbr_entrainement=20
learning_rate=0.0001

mnist_train_images=np.fromfile("mnist/train-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_train_labels=np.eye(10)[np.fromfile("mnist/train-labels-idx1-ubyte", dtype=np.uint8)[8:]]
mnist_test_images=np.fromfile("mnist/t10k-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_test_labels=np.eye(10)[np.fromfile("mnist/t10k-labels-idx1-ubyte", dtype=np.uint8)[8:]]

ph_images=tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32)
ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)

result=convolution(ph_images, 5, 32)
result=convolution(result, 5, 32)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result=convolution(result, 5, 128)
result=convolution(result, 5, 128)
result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result=tf.contrib.layers.flatten(result)

result=fc(result, 512)
result=tf.nn.relu(result)
result=fc(result, 10)
scso=tf.nn.softmax(result)

loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=result)
train=tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), tf.float32))

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]
    for id_entrainement in np.arange(nbr_entrainement):
        tab_accuracy_train=[]
        tab_accuracy_test=[]
        for batch in np.arange(0, len(mnist_train_images), taille_batch):
            s.run(train, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })
        for batch in np.arange(0, len(mnist_train_images), taille_batch):
            precision=s.run(accuracy, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })
            tab_accuracy_train.append(precision)
        for batch in np.arange(0, len(mnist_test_images), taille_batch):
            precision=s.run(accuracy, feed_dict={
                ph_images: mnist_test_images[batch:batch+taille_batch],
                ph_labels: mnist_test_labels[batch:batch+taille_batch]
            })
            tab_accuracy_test.append(precision)
        print("> Entrainement", id_entrainement)
        print("  train:", np.mean(tab_accuracy_train))
        tab_train.append(1-np.mean(tab_accuracy_train))
        print("  test :", np.mean(tab_accuracy_test))
        tab_test.append(1-np.mean(tab_accuracy_test))

    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_train, label="Train error")
    plot.plot(tab_test, label="Test error")
    plot.legend(loc="upper right")
    plot.show()
    
    resulat=s.run(scso, feed_dict={ph_images: mnist_test_images[0:taille_batch]})
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    for image in range(taille_batch):
        print("image", image)
        print("sortie du réseau:", resulat[image], np.argmax(resulat[image]))
        print("sortie attendue :", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
        cv2.imshow('image', mnist_test_images[image].reshape(28, 28))
        if cv2.waitKey()==ord('q'):
            break
