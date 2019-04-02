import cv2
import numpy as np
import tensorflow as tf

cap=cv2.VideoCapture(0)
np.set_printoptions(formatter={'float': '{:0.3f}'.format})
with tf.Session() as s:
    saver=tf.train.import_meta_graph('./mon_vgg/modele.meta')
    saver.restore(s, tf.train.latest_checkpoint('./mon_vgg/'))
    graph=tf.get_default_graph()
    images=graph.get_tensor_by_name("images:0")
    sortie=graph.get_tensor_by_name("sortie:0")
    is_training=graph.get_tensor_by_name("is_training:0")
    while True:
        ret, frame=cap.read()
        test=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        test=cv2.resize(test, (28, 28))
        for x in range(28):
            for y in range(28):
                if test[y][x]<110:
                    test[y][x]=1
                else:
                    test[y][x]=0
        cv2.imshow('image', cv2.resize(test, (120, 120))*255)
        prediction=s.run(sortie, feed_dict={images: [test.reshape(28, 28, 1)], is_training: False})
        print(prediction, np.argmax(prediction))
        if cv2.waitKey(20)&0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
