import cv2
import numpy as np
import common

video='autoroute.mp4'
image_fond="img-0.png"
color_infos=(0, 0, 255)

nbr_old=0
vehicule=0
seuil=10

fond=common.moyenne_image(video, 100)

cv2.imshow('fond', fond.astype(np.uint8))
cap=cv2.VideoCapture(video)

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    mask=common.calcul_mask(frame, fond, seuil)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    nbr=0
    for e in elements:
        ((x, y), rayon)=cv2.minEnclosingCircle(e)
        if rayon>20:
            cv2.circle(frame, (int(x), int(y)), 5, color_infos, 10)
            nbr+=1
    if nbr>nbr_old:
        vehicule+=1
    nbr_old=nbr
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}  Seuil: {:d}".format(fps, seuil), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_infos, 1)
    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('p'):
        seuil+=1
    if key==ord('m'):
        seuil-=1
    if key==ord('a'):
        for cpt in range(20):
            ret, frame=cap.read()

cap.release()
cv2.destroyAllWindows()
