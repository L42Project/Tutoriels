import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

def stl10(path):
    labels=['avion', 'oiseau', 'voiture', 'chat', 'cerf', 'chien', 'cheval', 'singe', 'bateau', 'camion']
    train_images=np.fromfile(path+"/train_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)
    train_labels=np.eye(10)[np.fromfile(path+"/train_y.bin", dtype=np.uint8)-1]
    train_images, train_labels=shuffle(train_images, train_labels)
    test_images=np.fromfile(path+"/test_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)
    test_labels=np.eye(10)[np.fromfile(path+"/test_y.bin", dtype=np.uint8)-1]
    return labels, train_images, train_labels, test_images, test_labels

def convolution(input, taille_noyau, nbr_noyau, stride, b_norm, f_activation, training):
    w_filtre=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(input.get_shape()[-1]), nbr_noyau)))
    b_filtre=np.zeros(nbr_noyau)
    result=tf.nn.conv2d(input, w_filtre, strides=[1, stride, stride, 1], padding='SAME')+b_filtre
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)
    return result
        
def fc(input, nbr_neurone, b_norm, f_activation, training):
    w=tf.Variable(tf.random.truncated_normal(shape=(int(input.get_shape()[-1]), nbr_neurone), dtype=tf.float32))
    b=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(input, w)+b
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)                        
    return result

def b_resnet_1(input, kernel, nbr_cc, reduce, training, dropout=None):
    if reduce is True:
        stride=2
        result2=convolution(input, 1, nbr_cc[-1], stride, True, tf.nn.relu, training)
    else:
        stride=1
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(input, 1, nbr_cc[-1], stride, True, tf.nn.relu, training)
        else:
            result2=input
    result=input
    for shift in range(len(kernel)-1):
        result=convolution(result, kernel[shift], nbr_cc[shift], stride, True, tf.nn.relu, training)
        stride=1
    result=convolution(result, kernel[len(kernel)-1], nbr_cc[len(kernel)-1], stride, True, None, training)
    result=result+result2
    result=tf.nn.relu(result)
    if dropout is not None:
        result=tf.layers.dropout(result, dropout)
    return result

def b_resnet_1M(input, kernel, nbr_cc, reduce, training, dropout=None):
    if reduce is True:
        result=tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        result2=tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(result2, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
    else:
        result=input
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(input, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
        else:
            result2=input
    for shift in range(len(kernel)-1):
        result=convolution(result, kernel[shift], nbr_cc[shift], 1, True, tf.nn.relu, training)
    result=convolution(result, kernel[len(kernel)-1], nbr_cc[len(kernel)-1], 1, True, None, training)
    result=result+result2
    result=tf.nn.relu(result)
    if dropout is not None:
        result=tf.layers.dropout(result, dropout)
    return result

def b_resnet_2(input, kernel, nbr_cc, reduce, training, dropout=None):
    if reduce is True:
        stride=2
        result2=convolution(input, 1, nbr_cc[-1], stride, True, tf.nn.relu, training)
    else:
        stride=1
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(input, 1, nbr_cc[-1], stride, True, tf.nn.relu, training)
        else:
            result2=input
    result=input
    for shift in range(len(kernel)-1):
        result=convolution(result, kernel[shift], nbr_cc[shift], stride, True, tf.nn.relu, training)
        stride=1
    result=convolution(result, kernel[len(kernel)-1], nbr_cc[len(kernel)-1], stride, False, None, training)
    result=result+result2
    result=tf.layers.batch_normalization(result, training=training)
    result=tf.nn.relu(result)
    if dropout is not None:
        result=tf.layers.dropout(result, dropout)
    return result

def b_resnet_2M(input, kernel, nbr_cc, reduce, training, dropout=None):
    if reduce is True:
        result=tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        result2=tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(result2, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
    else:
        result=input
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(input, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
        else:
            result2=input
    for shift in range(len(kernel)-1):
        result=convolution(result, kernel[shift], nbr_cc[shift], 1, True, tf.nn.relu, training)
    result=convolution(result, kernel[len(kernel)-1], nbr_cc[len(kernel)-1], 1, False, None, training)
    result=result+result2
    result=tf.layers.batch_normalization(result, training=training)
    result=tf.nn.relu(result)
    if dropout is not None:
        result=tf.layers.dropout(result, dropout)
    return result

def b_resnet_3(input, kernel, nbr_cc, reduce, training, dropout=None):
    if reduce is True:
        stride=2
        result2=convolution(input, 1, nbr_cc[-1], stride, True, tf.nn.relu, training)
    else:
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(input, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
        else:
            result2=input
        stride=1
    result=input
    for shift in range(len(kernel)-1):
        result=convolution(result, kernel[shift], nbr_cc[shift], stride, True, tf.nn.relu, training)
        stride=1
    shift=len(kernel)-1
    result=convolution(result, kernel[shift], nbr_cc[shift], stride, True, None, training)        
    result=result+result2
    if dropout is not None:
        result=tf.layers.dropout(result, dropout)
    return result

def b_resnet_3M(input, kernel, nbr_cc, reduce, training, dropout=None):
    if reduce is True:
        result =tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        result2=tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(result2, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
    else:
        result=convolution(input, kernel[0], nbr_cc[0], 1, True, tf.nn.relu, training)
        if nbr_cc[-1]!=int(input.get_shape()[-1]):
            result2=convolution(input, 1, nbr_cc[-1], 1, True, tf.nn.relu, training)
        else:
            result2=input
    for shift in range(1, len(kernel)):
        result=convolution(result, kernel[shift], nbr_cc[shift], 1, True, tf.nn.relu, training)
    result=result+result2
    if dropout is not None:
        result=tf.layers.dropout(result, dropout)
    return result

def resnet(nbr_classes, b_resnet, learning_rate):
    ph_images=tf.placeholder(shape=(None, 96, 96, 3), dtype=tf.float32)
    ph_labels=tf.placeholder(shape=(None, nbr_classes), dtype=tf.float32)
    ph_is_training=tf.placeholder_with_default(False, (), name='is_training')
    
    #result=convolution(ph_images, 7, 64, 2, True, tf.nn.relu, ph_is_training)
    result=convolution(ph_images, 5, 64, 1, True, tf.nn.relu, ph_is_training)
    result=tf.nn.max_pool(result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    result=b_resnet(result, [1, 3, 1], [64, 64, 256], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [64, 64, 256], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [64, 64, 256], False, ph_is_training, None)
    
    result=b_resnet(result, [1, 3, 1], [128, 128, 512], True, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [128, 128, 512], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [128, 128, 512], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [128, 128, 512], False, ph_is_training, None)
    
    result=b_resnet(result, [1, 3, 1], [256, 256, 1024], True, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [256, 256, 1024], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [256, 256, 1024], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [256, 256, 1024], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [256, 256, 1024], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [256, 256, 1024], False, ph_is_training, None)
    
    result=b_resnet(result, [1, 3, 1], [512, 512, 2048], True, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [512, 512, 2048], False, ph_is_training, None)
    result=b_resnet(result, [1, 3, 1], [512, 512, 2048], False, ph_is_training, None)
    taille=result.get_shape()[1]
    result=tf.nn.avg_pool(result, ksize=[1, taille, taille, 1], strides=[1, 1, 1, 1], padding='SAME')
    
    result=tf.contrib.layers.flatten(result)    
    result=fc(result, nbr_classes, False, None, ph_is_training)
    socs=tf.nn.softmax(result)
    
    loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=result)
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(socs, 1), tf.argmax(ph_labels, 1)), tf.float32))

    return ph_images, ph_labels, ph_is_training, socs, train, accuracy, tf.train.Saver()
