import cv2
import numpy as np
import tensorflow as tf
from L42Project import ia as LPia

mnist_test_images=np.fromfile("mnist/t10k-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_test_labels=np.eye(10)[np.fromfile("mnist/t10k-labels-idx1-ubyte", dtype=np.uint8)[8:]]

tf.reset_default_graph()
np.set_printoptions(formatter={'float': '{:0.3f}'.format})
with tf.Session() as s:
    saver=tf.train.import_meta_graph('./mon_vgg/modele.meta')
    saver.restore(s, tf.train.latest_checkpoint('./mon_vgg/'))
    graph=tf.get_default_graph()
    images=graph.get_tensor_by_name("images:0")
    sortie=graph.get_tensor_by_name("sortie:0")
    is_training=graph.get_tensor_by_name("is_training:0")
    while True:
        image=cv2.imread("/home/laurent/chiffre.png", cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image, (28, 28))
        image=image.reshape(28, 28, 1)/255
        test_images=[]
        test_images.append(image)
        test_images=np.asarray(test_images)
        #cv2.imshow('image', test_images[0])
        for i in mnist_test_images[0:10]:
        #for i in test_images:
            prediction=s.run(sortie, feed_dict={images: [i], is_training: False})
            print(prediction, np.argmax(prediction))
        #if cv2.waitKey()==ord('q'):
        #    break
        break
cv2.destroyAllWindows()    
