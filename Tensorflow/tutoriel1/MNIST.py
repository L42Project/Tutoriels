import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import cv2

nbr_ni=100
learning_rate=0.0001
taille_batch=100
nbr_entrainement=200

mnist_train_images=np.fromfile("mnist/train-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 784)/255
mnist_train_labels=np.eye(10)[np.fromfile("mnist/train-labels-idx1-ubyte", dtype=np.uint8)[8:]]
mnist_test_images=np.fromfile("mnist/t10k-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 784)/255
mnist_test_labels=np.eye(10)[np.fromfile("mnist/t10k-labels-idx1-ubyte", dtype=np.uint8)[8:]]

ph_images=tf.placeholder(shape=(None, 784), dtype=tf.float32)
ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)

wci=tf.Variable(tf.truncated_normal(shape=(784, nbr_ni)), dtype=tf.float32)
bci=tf.Variable(np.zeros(shape=(nbr_ni)), dtype=tf.float32)
sci=tf.matmul(ph_images, wci)+bci
sci=tf.nn.sigmoid(sci)

wcs=tf.Variable(tf.truncated_normal(shape=(nbr_ni, 10)), dtype=tf.float32)
bcs=tf.Variable(np.zeros(shape=(10)), dtype=tf.float32)
scs=tf.matmul(sci, wcs)+bcs
scso=tf.nn.softmax(scs)

loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=scs)
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), dtype=tf.float32))

with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    tab_acc_train=[]
    tab_acc_test=[]
    for id_entrainement in range(nbr_entrainement):
        print("ID entrainement", id_entrainement)
        for batch in range(0, len(mnist_train_images), taille_batch):
            s.run(train, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })

        tab_acc=[]
        for batch in range(0, len(mnist_train_images), taille_batch):
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })
            tab_acc.append(acc)
        print("accuracy train:", np.mean(tab_acc))
        tab_acc_train.append(1-np.mean(tab_acc))
        
        tab_acc=[]
        for batch in range(0, len(mnist_test_images), taille_batch):
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_test_images[batch:batch+taille_batch],
                ph_labels: mnist_test_labels[batch:batch+taille_batch]
            })
            tab_acc.append(acc)
        print("accuracy test :", np.mean(tab_acc))
        tab_acc_test.append(1-np.mean(tab_acc))
        
    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_acc_train, label="Train error")
    plot.plot(tab_acc_test, label="Test error")
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


    
