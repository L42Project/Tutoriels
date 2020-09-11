import cv2
import numpy as np
import glob

x_min=600
y_min=350
x_max=750
y_max=500

fichiers=glob.glob("poulailler*.jpg")
for fichier in fichiers:
    img=cv2.imread(fichier, 1)
    img_cible=img[y_min:y_max, x_min:x_max]
    edges=~cv2.Canny(img_cible, 30, 80)

    edges=cv2.erode(edges, None, iterations=3)
    edges=cv2.dilate(edges, None, iterations=2)

    presence=0
    elements=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for e in elements:
        c=cv2.contourArea(e)
        if c>150:
            presence=1

    if presence:
        cv2.putText(img, "OK", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "KO", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    cv2.imshow("image", img)
    cv2.imshow("image2", edges)
    cv2.waitKey()
