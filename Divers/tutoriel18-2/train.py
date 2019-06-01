import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

taille_batch=100
nbr_entrainement=100
nbr=2*42

def modif_image(image, seuil=1):
    b=np.random.normal(0, 1, (28, 28))
    a=image.copy()
    a[b>seuil]=255
    a[b<-seuil]=0
    return a

def convolution(input, taille_noyau, nbr_noyau, stride, b_norm=False, f_activation=None, training=False):
    w_filtre=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(input.get_shape()[-1]), nbr_noyau)))
    b_filtre=np.zeros(nbr_noyau)
    result=tf.nn.conv2d(input, w_filtre, strides=[1, stride, stride, 1], padding='SAME')+b_filtre
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)
    return result
        
def fc(input, nbr_neurone, b_norm=False, f_activation=None, training=False):
    w=tf.Variable(tf.random.truncated_normal(shape=(int(input.get_shape()[-1]), nbr_neurone), dtype=tf.float32))
    b=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(input, w)+b
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)                        
    return result

def ia(nbr_classes, size, couche, learning_rate=1E-3):
    ph_images=tf.placeholder(shape=(None, size, size, couche), dtype=tf.float32, name='entree')
    ph_labels=tf.placeholder(shape=(None, nbr_classes), dtype=tf.float32)
    ph_is_training=tf.placeholder_with_default(False, (), name='is_training')
    
    result=convolution(ph_images, 3, 64, 1, True, tf.nn.relu, ph_is_training)
    result=tf.layers.dropout(result, 0.3, training=ph_is_training)
    result=convolution(result, 3, 128, 1, True, tf.nn.relu, ph_is_training)
    result=tf.layers.dropout(result, 0.4, training=ph_is_training)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    result=tf.contrib.layers.flatten(result)
    
    result=fc(result, 128, True, tf.nn.relu, ph_is_training)
    result=tf.layers.dropout(result, 0.5, training=ph_is_training)
    result=fc(result, nbr_classes)
    socs=tf.nn.softmax(result, name="sortie")
    
    loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=result)
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(socs, 1), tf.argmax(ph_labels, 1)), tf.float32))

    return ph_images, ph_labels, ph_is_training, socs, train, accuracy, tf.train.Saver()

tab_images=[]
tab_labels=[]

for dir in ["/usr/share/fonts/truetype/ubuntu-font-family/", "/usr/share/fonts/truetype/freefont/"]:
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith("ttf"):
                print(root+"/"+file)
                for i in range(1, 10):
                    for cpt in range(nbr):
                        image=Image.new("L", (28, 28))
                        draw=ImageDraw.Draw(image)
                        font=ImageFont.truetype(root+"/"+file, np.random.randint(26, 32))
                        text="{:d}".format(i)
                        draw.text((np.random.randint(1, 10), np.random.randint(-4, 0)), text, font=font, fill=(255))
                        image=np.array(image).reshape(28, 28, 1)
                        tab_images.append(image)
                        tab_labels.append(np.eye(10)[i])                        
                        image_m=modif_image(image, 1.05+np.random.rand())
                        tab_images.append(image_m)
                        tab_labels.append(np.eye(10)[i])
                image=np.zeros((28, 28, 1))
                for cpt in range(3*nbr):
                    image_m=modif_image(image, 1.05+np.random.rand())
                    tab_images.append(image_m)
                    tab_labels.append(np.eye(10)[0])
                    
tab_images=np.array(tab_images)
tab_labels=np.array(tab_labels)

tab_images=tab_images/255

tab_images, tab_labels=shuffle(tab_images, tab_labels)

if False: # Changer en True si vous voulez voir les images générées
    for i in range(len(tab_images)):
        cv2.imshow('chiffre', tab_images[i].reshape(28, 28, 1))
        print(tab_labels[i], np.argmax(tab_labels[i]))
        if cv2.waitKey()&0xFF==ord('q'):
            break

print("Nbr:", len(tab_images))

train_images, test_images, train_labels, test_labels=train_test_split(tab_images, tab_labels, test_size=0.10)

images, labels, is_training, sortie, train, accuracy, saver=ia(10, 28, 1)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]
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
    saver.save(s, './mon_modele/modele')
