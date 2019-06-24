import tensorflow as tf
import os
import numpy as np
import cv2

width=200
height=125

dir='dataC/'

tab_color=[(255, 0, 0), (255, 0, 255)]
tab_label=['Voiture', 'Signalisation']
tab_value=[0.2, 0.2]
tab_surface=[500, 200]

with tf.Session() as s:
    saver=tf.train.import_meta_graph('./mon_modele/modele.meta')
    saver.restore(s, tf.train.latest_checkpoint('./mon_modele/'))
    graph=tf.get_default_graph()
    images=graph.get_tensor_by_name("entree:0")
    sortie=graph.get_tensor_by_name("sortie:0")
    l=os.listdir(dir+"CameraRGB/")
    l=sorted(l)
    for file in l:
        img=cv2.imread(dir+"CameraRGB/"+file)[0:500, :]
        prediction=s.run(sortie, feed_dict={images: [cv2.resize(img, (width, height))/255]})
        for m in range(prediction[0].shape[-1]):
            mask=np.zeros(shape=(prediction[0].shape[0], prediction[0].shape[1]))
            mask[prediction[0][:, :, m]>tab_value[m]]=1.
            mask=cv2.resize(mask, (4*width, 4*height))
            elements=cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for e in elements:
                if cv2.contourArea(e)>tab_surface[m]:
                    x,y,w,h = cv2.boundingRect(e)
                    cv2.putText(img, tab_label[m], (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, tab_color[m], 2)
                    cv2.rectangle(img, (x, y), (x+w, y+h), tab_color[m], 2)
        cv2.imshow("Resultat", img)
        key=cv2.waitKey()&0xFF
        if key==ord('q'):
            break
