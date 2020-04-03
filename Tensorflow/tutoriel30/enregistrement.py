import cv2
import numpy as np
import os
import config

saut=10

dir=config.dir_neg
#dir=config.dir_pos
os.makedirs(dir, exist_ok=True)

id=0
while os.path.isfile(dir+"image-{:d}.png".format(id)):
    id+=1
id*=saut

cap=cv2.VideoCapture(0)
width=int(cap.get(3))
enregistre=0
while True:
    ret, frame=cap.read()

    if enregistre:
        if not id%saut:
            id_=int(id/saut)
            fichier=dir+"image-{:d}.png".format(id_)
            print("Création du fichier", fichier)
            cv2.imwrite(fichier, frame)
        id+=1

    cv2.rectangle(frame, (0, 0), (width, 30), (100, 100, 100), cv2.FILLED)
    cv2.putText(frame, "[e] enregistrement  repertoire: {}  [q] quitter".format(dir), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 0)
    if enregistre:
        cv2.circle(frame, (width-20, 15), 5, (0, 0, 255), 8)

    cv2.imshow('Camera', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('e'):
        enregistre=not enregistre
    if key==ord('q'):
        quit()
