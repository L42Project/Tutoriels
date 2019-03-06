import tensorflow as tf
import numpy as np

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

def vggnet(learning_rate=0.01, momentum=0.99):
    ph_images=tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='images')
    ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)
    ph_is_training=tf.placeholder_with_default(False, (), name='is_training')
    
    result=convolution(ph_images, 3, 64)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 64)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    result=convolution(result, 3, 128)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 128)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    result=convolution(result, 3, 256)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 256)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 256)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    result=convolution(result, 3, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    result=convolution(result, 3, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=convolution(result, 3, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    result=tf.contrib.layers.flatten(result)
    
    result=fc(result, 512)
    result=tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)
    result=tf.nn.relu(result)
    result=fc(result, 10)
    socs=tf.nn.softmax(result, name="sortie")
    
    loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=result)
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(socs, 1), tf.argmax(ph_labels, 1)), tf.float32))

    return ph_images, ph_labels, ph_is_training, socs, train, accuracy, tf.train.Saver()

