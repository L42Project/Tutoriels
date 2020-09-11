import cv2
import numpy as np
import glob

fichiers=glob.glob("*.jfif")
for fichier in fichiers:
    img=cv2.imread(fichier, 1)
    edges=~cv2.Canny(img, 80, 150)

    edges=cv2.erode(edges, None, iterations=3)
    edges=cv2.dilate(edges, None, iterations=2)

    presence=0
    elements=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for e in elements:
        c=cv2.contourArea(e)
        if c>150:
            cv2.drawContours(img, [e], 0, (255, 0, 0), 2)
            presence=1

    if presence:
        cv2.putText(img, "OK", (50, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "KO", (50, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("image", img)
    cv2.imshow("image2", edges)
    cv2.waitKey()
