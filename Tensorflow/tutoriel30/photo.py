import cv2
import numpy as np
import os
import config

os.makedirs(config.dir_pos, exist_ok=True)
id_pos=0
while os.path.isfile(config.dir_pos+"image-{:d}.png".format(id_pos)):
    id_pos+=1

os.makedirs(config.dir_neg, exist_ok=True)
id_neg=0
while os.path.isfile(config.dir_neg+"image-{:d}.png".format(id_neg)):
    id_neg+=1

cap=cv2.VideoCapture(0)
width=int(cap.get(3))

while True:
    ret, frame=cap.read()

    cv2.rectangle(frame, (0, 0), (width, 30), (100, 100, 100), cv2.FILLED)
    cv2.putText(frame, "[p] photo positive    [n] photo negative    [q] quitter", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow('Camera', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('p'):
        fichier=config.dir_pos+"image-{:d}.png".format(id_pos)
        print("Création du fichier", fichier)
        cv2.imwrite(fichier, frame)
        id_pos+=1
    if key==ord('n'):
        fichier=config.dir_neg+"image-{:d}.png".format(id_neg)
        print("Création du fichier", fichier)
        cv2.imwrite(fichier, frame)
        id_neg+=1
    if key==ord('q'):
        quit()
