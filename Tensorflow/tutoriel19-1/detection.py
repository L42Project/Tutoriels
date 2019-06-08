import tensorflow as tf
import os 
import numpy as np
import cv2

width=160
height=120

dir='dataE/'

with tf.Session() as s:
    saver=tf.train.import_meta_graph('./mon_modele/modele.meta')
    saver.restore(s, tf.train.latest_checkpoint('./mon_modele/'))
    graph=tf.get_default_graph()
    images=graph.get_tensor_by_name("entree:0")
    sortie=graph.get_tensor_by_name("sortie:0")

    for file in os.listdir(dir+'CameraRGB/'):
        img=cv2.resize(cv2.imread(dir+'CameraRGB/'+file), (width, height))/255                
        cv2.imshow("image", img)
        m=cv2.resize(cv2.imread(dir+'CameraSeg/'+file)[:,:,2], (width, height))
        m[m==7]=255
        m[m!=255]=0
        cv2.imshow("mask 7", m)
        m=cv2.resize(cv2.imread(dir+'CameraSeg/'+file)[:,:,2], (width, height))
        m[m==9]=255
        m[m!=255]=0
        cv2.imshow("mask 9", m)
        prediction=s.run(sortie, feed_dict={images:[img]})
        cv2.imshow("mask prediction 7", prediction[0][:,:,0])
        cv2.imshow("mask prediction 9", prediction[0][:,:,1])
        if cv2.waitKey()&0xFF==ord('q'):
            break
            
            
