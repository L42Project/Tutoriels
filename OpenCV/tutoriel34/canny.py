import cv2
import numpy as np
import glob

fichiers=glob.glob("*.jfif")
for fichier in fichiers:
    img=cv2.imread(fichier, 1)
    edges=~cv2.Canny(img, 80, 150)

    edges=cv2.erode(edges, None, iterations=3)
    edges=cv2.dilate(edges, None, iterations=2)

    cv2.imshow("image", img)
    cv2.imshow("edges", edges)
    cv2.waitKey()
