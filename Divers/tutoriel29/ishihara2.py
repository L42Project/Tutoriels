import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import cv2
import sys
import glob

min_clusters=2
max_clusters=4
ESPACES=["YCrCb", "HSV", "LAB"]
COUCHES=[[1], [0, 2], [1, 2]]
size=200

for image in glob.glob('.\images\*.png'):
    print("Image: {} ".format(image), end='')
    tab=np.zeros([(len(ESPACES))*size, (max_clusters-min_clusters+1)*size], dtype=np.float32)
    img=cv2.imread(image)
    cv2.imshow("image", cv2.resize(img, (2*size, 2*size)))
    img=cv2.resize(img, (size, size))

    for index in range(len(ESPACES)):
        img2=cv2.cvtColor(img, eval("cv2.COLOR_BGR2"+ESPACES[index]))
        X=img2[:, :, COUCHES[index]].reshape(img2.shape[0]*img2.shape[1], len(COUCHES[index]))
        for k in range(min_clusters, max_clusters+1):
            sys.stdout.write('.')
            sys.stdout.flush()
            kmeans=KMeans(n_clusters=k)
            pred=kmeans.fit_predict(X)
            pred=pred.reshape(img2.shape[0], img2.shape[1])
            pred=pred/(k-1)
            tab[index*size:(index+1)*size, (k-min_clusters)*size:(k-min_clusters)*size+size]=pred
    sys.stdout.write('\n')
    cv2.imshow("kmeans", tab)
    if cv2.waitKey()&0xFF==ord('q'):
        break
