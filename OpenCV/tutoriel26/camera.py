import os
import sys
import time
import numpy as np
import cv2

# mode global=1 local=0
mode=1
label_mode=["local", "global"]
kernel_blur=3
seuil=15
seuil_nbr_pixel=5000
dir_videos="d:\\enregistrements\\"

couleur_fond      =(100, 100, 100)
couleur_infos     =(255, 255, 255)
couleur_alarme    =(  0,   0, 255)
couleur_fin_alarme=(  0, 128, 255)

cap=cv2.VideoCapture(1)
ret, originale=cap.read()
if ret is False:
    quit()

hauteur, largeur, nbr_couche=originale.shape
tab_hauteur=50

originale=cv2.cvtColor(originale, cv2.COLOR_BGR2GRAY)
originale=cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)

kernel_dilate=np.ones((3, 3), np.uint8)
tab=np.zeros((largeur), dtype=np.int32)
image_finale=np.zeros((hauteur+tab_hauteur, largeur, nbr_couche), dtype=np.uint8)
fichier_video=None

if not os.path.isdir(dir_videos):
    os.mkdir(dir_videos)

fin_mouvement=40
cpt_fin_mouvement=0

while True:
    alarme=0
    ret, frame=cap.read()
    if ret is False:
        quit()
    image_finale[:hauteur, :, :]=frame
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur=cv2.GaussianBlur(gray, (kernel_blur, kernel_blur), 0)
    mask=cv2.absdiff(originale, gray_blur)
    mask=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)[1]
    if mode==0:
        mask=cv2.dilate(mask, kernel_dilate, iterations=2)
        contours, hierarchy=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            c=max(contours, key=cv2.contourArea)
            nbr_pixel=int(cv2.contourArea(c))
            couleur=(0, 0, 255) if nbr_pixel>seuil_nbr_pixel else (0, 255, 0)
            cv2.drawContours(image_finale, [c], 0, couleur, 3)
            if nbr_pixel>seuil_nbr_pixel:
                alarme=1
    else:
        nbr_pixel=int(np.sum(mask)/255)
        if nbr_pixel>seuil_nbr_pixel:
            alarme=1
    cv2.rectangle(image_finale, (0, 0), (image_finale.shape[1], 30), couleur_fond, cv2.FILLED)
    if alarme:
        cv2.putText(image_finale, "ALARME", (image_finale.shape[1]-80, 20), cv2.FONT_HERSHEY_PLAIN, 1, couleur_alarme, 2)
        if fichier_video is None:
            fichier_video=dir_videos+time.strftime("%Y_%m_%d_%H_%M_%S")+".avi"
            video=cv2.VideoWriter(fichier_video, cv2.VideoWriter_fourcc(*'DIVX'), 15, (largeur, hauteur))
        video.write(frame)
        cpt_fin_mouvement=fin_mouvement
    else:
        cpt_fin_mouvement=cpt_fin_mouvement-1
        if fichier_video is not None:
            if cpt_fin_mouvement==0:
                video.release()
                fichier_video=None
            else:
                cv2.putText(image_finale, "ALARME", (image_finale.shape[1]-80, 20), cv2.FONT_HERSHEY_PLAIN, 1, couleur_fin_alarme, 2)
                video.write(frame)
    txt="[p|m]Nbr pixel: {:d}   [o]Mode:{}   nbr pixel: {:06d}".format(seuil_nbr_pixel, label_mode[mode], nbr_pixel)
    tab=np.roll(tab, 1)
    tab[0]=int(nbr_pixel/300)
    tab_image=np.full((tab_hauteur, largeur, 3), couleur_fond, dtype=np.float32)
    ligne_seuil=int(seuil_nbr_pixel/300)
    for i in range(largeur):
        couleur=(0, 0, 255) if tab[i]>ligne_seuil else (0, 255, 0)
        cv2.line(tab_image, (i, tab_hauteur), (i, tab_hauteur-tab[i]), couleur, 1)
    cv2.line(tab_image, (0, tab_hauteur-ligne_seuil), (largeur, tab_hauteur-ligne_seuil), (0, 0, 255), 1)
    image_finale[hauteur:, :, :]=tab_image
    cv2.putText(image_finale, txt, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, couleur_infos, 2)
    originale=gray_blur
    cv2.imshow("Camera", image_finale)
    cv2.imshow("Mask", mask)
    key=cv2.waitKey(50)&0xFF
    if key==ord('q'):
        break
    if key==ord('p'):
        seuil_nbr_pixel+=100
    if key==ord('m'):
        seuil_nbr_pixel=max(100, seuil_nbr_pixel-100)
    if key==ord('o'):
        mode=not mode

cap.release()
cv2.destroyAllWindows()
