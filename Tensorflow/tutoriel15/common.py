import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

def stl10(path):
    labels=['avion', 'oiseau', 'voiture', 'chat', 'cerf', 'chien', 'cheval', 'singe', 'bateau', 'camion']
    train_images=np.fromfile(path+"/train_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)
    train_labels=np.eye(10)[np.fromfile(path+"/train_y.bin", dtype=np.uint8)-1]
    train_images, train_labels=shuffle(train_images, train_labels)
    test_images=np.fromfile(path+"/test_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 2, 3, 1)
    test_labels=np.eye(10)[np.fromfile(path+"/test_y.bin", dtype=np.uint8)-1]
    return labels, train_images, train_labels, test_images, test_labels

def couche_convolution(couche_prec, taille_noyau, nbr_noyau, stride, b_norm, f_activation, training):
    w_filtre=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_prec.get_shape()[-1]), nbr_noyau)))
    b_filtre=np.zeros(nbr_noyau)
    result=tf.nn.conv2d(couche_prec, w_filtre, strides=[1, stride, stride, 1], padding='SAME')+b_filtre
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)
    return result
        
def couche_fc(couche_prec, nbr_neurone, b_norm, f_activation, training):
    w=tf.Variable(tf.random.truncated_normal(shape=(int(couche_prec.get_shape()[-1]), nbr_neurone), dtype=tf.float32))
    b=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(couche_prec, w)+b
    if b_norm is True:
        result=tf.layers.batch_normalization(result, training=training)
    if f_activation is not None:
        result=f_activation(result)                        
    return result

def b_inception_v1(input, nbr_1, nbr_3r, nbr_3, nbr_5r, nbr_5, nbr_pool, training):
    result1=couche_convolution(input, 1, nbr_1, 1, True, tf.nn.relu, training)

    result2=couche_convolution(input, 1, nbr_3r, 1, True, tf.nn.relu, training)
    result2=couche_convolution(result2, 3, nbr_3, 1, True, tf.nn.relu, training)

    result3=couche_convolution(input, 1, nbr_5r, 1, True, tf.nn.relu, training)
    result3=couche_convolution(result3, 5, nbr_5, 1, True, tf.nn.relu, training)

    result4=tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    result4=couche_convolution(result4, 1, nbr_pool, 1, True, tf.nn.relu, training)

    result=tf.concat([result1, result2, result3, result4], 3)
    print(result)
    return result

def aux(input, training, nbr_classes):
    result=tf.nn.avg_pool(input, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID')
    result=couche_convolution(result, 1, 128, 1, True, tf.nn.relu, training)
    result=tf.contrib.layers.flatten(result)
    result=couche_fc(result, 1000, True, tf.nn.relu, training)
    result=tf.layers.dropout(result, 0.7, training=training)
    result=couche_fc(result, nbr_classes,  False, None, training)
    return result

def inception_v1(nbr_classes):
    ph_images=tf.placeholder(shape=(None, 96, 96, 3), dtype=tf.float32)
    ph_labels=tf.placeholder(shape=(None, nbr_classes), dtype=tf.float32)
    ph_is_training=tf.placeholder_with_default(False, (), name='is_training')
    ph_learning_rate=tf.placeholder(dtype=tf.float32)

    result=couche_convolution(ph_images, 5, 64, 2, True, tf.nn.relu, ph_is_training)
    result=tf.nn.max_pool(result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    #result=couche_convolution(result, 3, 64, 2, True, tf.nn.relu, ph_is_training)
    result=couche_convolution(result, 3, 192, 1, True, tf.nn.relu, ph_is_training)
    result=tf.nn.max_pool(result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    result=b_inception_v1(result, 64, 96, 128, 16, 32, 32, ph_is_training)
    result=b_inception_v1(result, 128, 128, 192, 32, 96, 64, ph_is_training)
    result=tf.nn.max_pool(result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    result=b_inception_v1(result, 192, 96, 208, 16, 48, 64, ph_is_training)
    
    aux1=aux(result, ph_is_training, nbr_classes)
    
    result=b_inception_v1(result, 160, 112, 224, 24, 64, 64, ph_is_training)
    result=b_inception_v1(result, 128, 128, 256, 24, 64, 64, ph_is_training)
    result=b_inception_v1(result, 112, 144, 288, 32, 64, 64, ph_is_training)
    
    aux2=aux(result, ph_is_training, nbr_classes)
    
    result=b_inception_v1(result, 256, 160, 320, 32, 128, 128, ph_is_training)
    result=tf.nn.max_pool(result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    result=b_inception_v1(result, 256, 160, 320, 32, 128, 128, ph_is_training)
    result=b_inception_v1(result, 384, 192, 384, 48, 128, 128, ph_is_training)
    taille=result.get_shape()[1]
    result=tf.nn.avg_pool(result, ksize=[1, taille, taille, 1], strides=[1, 1, 1, 1], padding='SAME')
    
    result=tf.contrib.layers.flatten(result)
    
    result=couche_fc(result, 1000, True, tf.nn.relu, ph_is_training)
    result=tf.layers.dropout(result, 0.4, training=ph_is_training)
    result=couche_fc(result, nbr_classes, False, None, ph_is_training)
    socs=tf.nn.softmax(result)
    
    loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=result)+\
          0.3*tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=aux1)+\
          0.3*tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=aux2)

    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train=tf.train.RMSPropOptimizer(ph_learning_rate).minimize(loss)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(socs, 1), tf.argmax(ph_labels, 1)), tf.float32))

    return ph_images, ph_labels, ph_is_training, ph_learning_rate, socs, train, accuracy, tf.train.Saver()
