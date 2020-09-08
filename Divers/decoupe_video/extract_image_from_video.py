import cv2
import os

film='Le_chemin_du_passe.mp4'

if not os.path.exists(film):
    quit("Le film n'existe pas")

nom_film=film.split('.')[0]

cap=cv2.VideoCapture(film)

if not os.path.isdir(nom_film):
    os.mkdir(nom_film)

id=0
while True:
    print("#", end="", flush=True)
    for cpt in range(500):
        ret, frame=cap.read()
        if frame is None:
            print("")
            cap.release()
            quit()
    cv2.imwrite("{}/{}-{:d}.png".format(nom_film, nom_film, id), frame)
    id+=1
