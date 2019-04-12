import cv2
import numpy as np
import common

color_infos=(0, 0, 255)

ymin=315
ymax=360

xmin1=110
xmax1=190

xmin2=250
xmax2=330

xmin3=380
xmax3=460

video='autoroute.mp4'

vehicule1=0
vehicule2=0
vehicule3=0
seuil=10
seuil2=100

fond=common.moyenne_image(video, 500)
fond=fond[ymin:ymax, xmin1:xmax3]
cv2.imshow('fond', fond.astype(np.uint8))
fond=fond.astype(np.int32)
cap=cv2.VideoCapture(video)

def calcul_mean(image):
    height, width=image.shape
    s=0
    for y in range(height):
        for x in range(width):
            s+=image[y][x]
    return s/(height*width)

old_1=0
old_2=0
old_3=0
while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    mask=common.calcul_mask(frame[ymin:ymax, xmin1:xmax3], fond, seuil)

    if calcul_mean(mask[0:ymax-ymin, 0:xmax1-xmin1])> seuil2:
        if old_1==0:
            vehicule1+=1
            old_1=1
    else:
        old_1=0

    if calcul_mean(mask[0:ymax-ymin, xmin2-xmin1:xmax2-xmin1])> seuil2:
        if old_2==0:
            vehicule2+=1
            old_2=1
    else:
        old_2=0

    if calcul_mean(mask[0:ymax-ymin, xmin3-xmin1:xmax3-xmin1])> seuil2:
        if old_3==0:
            vehicule3+=1
            old_3=1
    else:
        old_3=0

    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}  Seuil: {:d}".format(fps, seuil), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_infos, 1)
    cv2.putText(frame, "{:04d} {:04d} {:04d}".format(vehicule1, vehicule2, vehicule3), (xmin1, ymin-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    cv2.rectangle(frame, (xmin1, ymin), (xmax1, ymax), (0, 0, 255) if old_1 else (255, 0, 0), 3)
    cv2.rectangle(frame, (xmin2, ymin), (xmax2, ymax), (0, 0, 255) if old_2 else (255, 0, 0), 3)
    cv2.rectangle(frame, (xmin3, ymin), (xmax3, ymax), (0, 0, 255) if old_3 else (255, 0, 0), 3)
    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('p'):
        seuil+=1
    if key==ord('m'):
        seuil-=1

cap.release()
cv2.destroyAllWindows()
