import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import cv2
import glob

k=2
ESPACE="HSV"
CH=[0, 2]
size=200

for image in glob.glob('.\images\*.png'):
    print("Image:", image)

    # Lecture et affichage de l'image
    img=cv2.imread(image)
    img=cv2.resize(img, (size, size))
    cv2.imshow("image", img)

    # Changement d'espace colorimétrique
    img=cv2.cvtColor(img, eval("cv2.COLOR_BGR2"+ESPACE))
    X=img[:, :, CH].reshape(img.shape[0]*img.shape[1], len(CH))

    # Graph 2D des couches A et B
    if len(CH)==2:
        plt.scatter(X[:,0], X[:,1], s=3)#, marker='+')
        plt.show()

    # Algorithme K moyennes
    kmeans=KMeans(n_clusters=k)
    #kmeans.fit(X)
    pred=kmeans.fit_predict(X)

    # Graph 2D des couches A et B après utilisation de l'algorithme K moyennes
    if len(CH)==2:
        plt.scatter(X[:,0], X[:,1], c=pred, s=3) #10, marker='+')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red')
        plt.show()

    # Affichage du résultat
    pred=pred.reshape(img.shape[0], img.shape[1])
    pred=pred/(k-1)
    cv2.imshow("kmeans", pred)

    if cv2.waitKey()&0xFF==ord('q'):
        break
