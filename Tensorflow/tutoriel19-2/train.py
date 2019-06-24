import cv2
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

dir_img="CameraRGB/"
dir_mask="CameraSeg/"

width=200
height=125

taille_batch=50
nbr_entrainement=100

def crop(tensor1, tensor2):
    offsets=(0, (int(tensor1.get_shape()[1])-int(tensor2.get_shape()[1]))//2, (int(tensor1.get_shape()[2])-int(tensor2.get_shape()[2]))//2, 0)
    size=(-1, int(tensor2.get_shape()[1]), int(tensor2.get_shape()[2]), -1)
    return tf.slice(tensor1, offsets, size)

def convolution(input, taille_noyau, nbr_cc, stride, b_norm=False, f_activation=None, training=False, padding='SAME'):
    w=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(input.get_shape()[-1]), nbr_cc)))
    b=np.zeros(nbr_cc)
    result=tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)
    result=tf.nn.bias_add(result, b)
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)
    return result

def deconvolution(input, taille_noyau, nbr_cc, stride, b_norm=False, f_activation=None, training=False, padding='SAME'):
    w=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, nbr_cc, int(input.get_shape()[-1]))))
    b=np.zeros(nbr_cc)
    if padding == 'VALID':
        out_h=(int(input.get_shape()[1])-1)*stride+taille_noyau
        out_w=(int(input.get_shape()[2])-1)*stride+taille_noyau
    elif padding == 'SAME':
        out_h=(int(input.get_shape()[1])-1)*stride+1
        out_w=(int(input.get_shape()[2])-1)*stride+1
    else:
        quit("erreur padding")
    b_size=tf.shape(input)[0]
    result=tf.nn.conv2d_transpose(input, w, output_shape=[b_size, out_h, out_w, nbr_cc], strides=[1, stride, stride, 1], padding=padding)
    result=tf.nn.bias_add(result, b)
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)
    return result

def unet(nbr_mask, size, padding='SAME', learning_rate=1E-3):
    ph_images=tf.placeholder(shape=(None, size[0], size[1], size[2]), dtype=tf.float32, name='entree')
    ph_masks=tf.placeholder(shape=(None, size[0], size[1], nbr_mask), dtype=tf.float32)
    ph_is_training=tf.placeholder_with_default(False, (), name='is_training')

    result=convolution(ph_images, 3, 32, 1, True, tf.nn.relu, ph_is_training, padding)
    c1=convolution(result, 3, 32, 1, True, tf.nn.relu, ph_is_training, padding)
    result=tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    result=convolution(result, 3, 64, 1, True, tf.nn.relu, ph_is_training, padding)
    c2=convolution(result, 3, 64, 1, True, tf.nn.relu, ph_is_training, padding)
    result=tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    result=convolution(result, 3, 128, 1, True, tf.nn.relu, ph_is_training, padding)
    c3=convolution(result, 3, 128, 1, True, tf.nn.relu, ph_is_training, padding)
    result=tf.nn.max_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    result=convolution(result, 3, 256, 1, True, tf.nn.relu, ph_is_training, padding)
    result=convolution(result, 3, 256, 1, True, tf.nn.relu, ph_is_training, padding)

    d3=deconvolution(result, 3, 256, 2, True, tf.nn.relu, ph_is_training, padding)
    c3=crop(c3, d3)
    result=tf.concat((d3, c3), axis=3)
    
    result=convolution(result, 3, 128, 1, True, tf.nn.relu, ph_is_training, padding)
    result=convolution(result, 3, 128, 1, True, tf.nn.relu, ph_is_training, padding)

    d2=deconvolution(result, 3, 128, 2, True, tf.nn.relu, ph_is_training, padding)
    c2=crop(c2, d2)
    result=tf.concat((d2, c2), axis=3)

    result=convolution(result, 3, 64, 1, True, tf.nn.relu, ph_is_training, padding)
    result=convolution(result, 3, 64, 1, True, tf.nn.relu, ph_is_training, padding)

    d1=deconvolution(result, 3, 64, 2, True, tf.nn.relu, ph_is_training, padding)
    c1=crop(c1, d1)
    result=tf.concat((d1, c1), axis=3)

    result=convolution(result, 3, 32, 1, True, tf.nn.relu, ph_is_training, padding)
    result=convolution(result, 3, 32, 1, True, tf.nn.relu, ph_is_training, padding)

    result=convolution(result, 1, nbr_mask, 1, False, None, ph_is_training, padding)
    result=tf.image.resize_images(result, (ph_masks.get_shape()[1], ph_masks.get_shape()[2]))    
    mask=tf.nn.sigmoid(result, name="sortie")

    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_masks, logits=result))
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.round(mask), ph_masks), tf.float32))
        
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return ph_images, ph_masks, ph_is_training, mask, train, accuracy, tf.train.Saver()

ph_images, ph_masks, ph_is_training, mask, train, accuracy, saver=unet(2, (height, width, 3), 'VALID')

tab_img=[]
tab_mask=[]
for dir in ['../tutoriel19-1/dataA/', '../tutoriel19-1/dataB/', '../tutoriel19-1/dataC/', '../tutoriel19-1/dataD/', '../tutoriel19-1/dataE/']:
    for file in os.listdir(dir+dir_img):
        tab_img.append(cv2.resize(cv2.imread(dir+dir_img+file)[0:500, 0:800], (width, height))/255)
        img_mask=cv2.resize(cv2.imread(dir+dir_mask+file)[0:500, 0:800], (width, height))[:,:,2]
        img_mask_result=np.zeros(shape=(height, width, 2), dtype=np.float32)
        img_mask_result[:,:,0][img_mask==10]=1.
        img_mask_result[:,:,1][img_mask==12]=1.
        tab_mask.append(img_mask_result)

tab_img=np.array(tab_img)
tab_mask=np.array(tab_mask)

train_images, test_images, train_labels, test_labels=train_test_split(tab_img, tab_mask, test_size=.05)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    tab_train=[]
    tab_test=[]
    for id_entrainement in np.arange(nbr_entrainement):
        print("> Entrainement", id_entrainement)
        for batch in np.arange(0, len(train_images), taille_batch):
            s.run(train, feed_dict={
                ph_images: train_images[batch:batch+taille_batch],
                ph_masks: train_labels[batch:batch+taille_batch],
                ph_is_training: True
            })
        print("  entrainement OK")
        tab_accuracy_train=[]
        for batch in np.arange(0, len(train_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                ph_images: train_images[batch:batch+taille_batch],
                ph_masks: train_labels[batch:batch+taille_batch]
            })
            tab_accuracy_train.append(p)
        print("  train:", np.mean(tab_accuracy_train))
        tab_accuracy_test=[]
        for batch in np.arange(0, len(test_images), taille_batch):
            p=s.run(accuracy, feed_dict={
                ph_images: test_images[batch:batch+taille_batch],
                ph_masks: test_labels[batch:batch+taille_batch]
            })
            tab_accuracy_test.append(p)
        print("  test :", np.mean(tab_accuracy_test))
        tab_train.append(1-np.mean(tab_accuracy_train))
        tab_test.append(1-np.mean(tab_accuracy_test))
    saver.save(s, './mon_modele/modele')
