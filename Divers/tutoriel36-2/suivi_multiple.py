import cv2
import numpy as np
from numpy import genfromtxt
from KalmanFilter import KalmanFilter
import math
import os
import glob

datasets="2DMOT2015Labels/train"
dataset="PETS09-S2L1"

distance_mini=500
rectangle=0
trace=0

dir_images=datasets+"/"+dataset+"/img1/"
fichier_label=datasets+"/"+dataset+"/gt/gt.txt"

if not os.path.exists(dir_images):
    print("Le repertoire n'existe pas ...", dir_images)
    quit()

if not os.path.exists(fichier_label):
    print("Le fichier de label n'existe pas ...", fichier)
    quit()

objets_points=[]
objets_id=[]
objets_KF=[]
objets_historique=[]

def distance(point, liste_points):
    distances=[]
    for p in liste_points:
        distances.append(np.sum(np.power(p-np.expand_dims(point, axis=-1), 2)))
    return distances

def trace_historique(tab_points, longueur, couleur=(0, 255, 255)):
    historique=np.array(tab_points)
    nbr_point=len(historique)
    longueur=min(nbr_point, longueur)
    for i in range(nbr_point-1, nbr_point-longueur, -1):
        cv2.line(frame,
                 (historique[i-1, 0], historique[i-1, 1]),
                 (historique[i, 0], historique[i, 1]),
                 couleur,
                 2)

data=genfromtxt(fichier_label, delimiter=',')
id_frame=0
id_objet=0

start=0
for image in glob.glob(dir_images+"*.jpg"):
    frame=cv2.imread(image)

    # Prediction de l'ensemble des objets + affichage
    for id_obj in range(len(objets_points)):
        etat=objets_KF[id_obj].predict()
        etat=np.array(etat, dtype=np.int32)
        objets_points[id_obj]=np.array([etat[0], etat[1], etat[4], etat[5]])
        objets_historique[id_obj].append([etat[0], etat[1]])
        cv2.circle(frame, (etat[0], etat[1]), 5, (0, 0, 255), 2)
        if rectangle:
            cv2.rectangle(frame,
                          (int(etat[0]-etat[4]/2), int(etat[1]-etat[5]/2)),
                          (int(etat[0]+etat[4]/2), int(etat[1]+etat[5]/2)),
                          (0, 0, 255),
                          2)
        cv2.arrowedLine(frame,
                        (etat[0], etat[1]),
                        (etat[0]+3*etat[2], etat[1]+3*etat[3]),
                        color=(0, 0, 255),
                        thickness=2,
                        tipLength=0.2)
        cv2.putText(frame,
                    "ID{:d}".format(objets_id[id_obj]),
                    (int(etat[0]-etat[4]/2), int(etat[1]-etat[5]/2)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 0),
                    2)

        if trace:
            trace_historique(objets_historique[id_obj], 42)

        # Permet de suivre l'ID 0 avec une flèche
        if objets_id[id_obj]==0:
            cv2.arrowedLine(frame,
                            (etat[0], int(etat[1]-etat[5]/2-80)),
                            (etat[0], int(etat[1]-etat[5]/2-30)),
                            color=(0, 0, 255),
                            thickness=5,
                            tipLength=0.2)

    # Récupération des objets de la frame concernée
    mask=data[:, 0]==id_frame

    # Affichage des données (rectangle) du detecteur
    points=[]
    for d in data[mask, :]:
        #if np.random.randint(2):
        if rectangle:
            cv2.rectangle(frame, (int(d[2]), int(d[3])), (int(d[2]+d[4]), int(d[3]+d[5])), (0, 255, 0), 2)
        xm=int(d[2]+d[4]/2)
        ym=int(d[3]+d[5]/2)
        cv2.circle(frame, (xm, ym), 2, (0, 255, 0), 2)
        points.append([xm, ym, int(d[4]), int(d[5])])

    # calcul des distances
    nouveaux_objets=np.ones((len(points)))
    tab_distances=[]
    if len(objets_points):
        for point_id in range(len(points)):
            distances=distance(points[point_id], objets_points)
            tab_distances.append(distances)

        tab_distances=np.array(tab_distances)
        sorted_distances=np.sort(tab_distances, axis=None)

        for d in sorted_distances:
            if d>distance_mini:
                break
            id1, id2=np.where(tab_distances==d)
            if not len(id1) or not len(id2):
                continue
            tab_distances[id1, :]=distance_mini+1
            tab_distances[:, id2]=distance_mini+1
            objets_KF[id2[0]].update(np.expand_dims(points[id1[0]], axis=-1))
            nouveaux_objets[id1]=0

    # Création du filtre de Kalman pour les nouveaux objets
    for point_id in range(len(points)):
        if nouveaux_objets[point_id]:
            print("NOUVEAU", points[point_id])
            objets_points.append(points[point_id])
            objets_KF.append(KalmanFilter(0.5, [points[point_id][0], points[point_id][1]], [points[point_id][2], points[point_id][3]]))
            objets_id.append(id_objet)
            objets_historique.append([])
            id_objet+=1

    # Nettoyage ...
    tab_id=[]
    for id_point in range(len(objets_points)):
        if int(objets_points[id_point][0])<-100 or \
        int(objets_points[id_point][1])<-100 or \
            objets_points[id_point][0]>frame.shape[1]+100 or \
            objets_points[id_point][1]>frame.shape[0]+100:
            print("SUPPRESSION", objets_points[id_point])
            tab_id.append(id_point)

    for index in sorted(tab_id, reverse=True):
        del objets_points[index]
        del objets_KF[index]
        del objets_id[index]
        del objets_historique[index]

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (100, 100, 100), cv2.FILLED)
    message="Frame: {:03d}  Nbr personne: {:d}  nbr filtre: {:d}   [r]Rectangle: {:3}  [t]Trace: {:3}".format(id_frame,
                                                                                                            len(points),
                                                                                                            len(objets_points),
                                                                                                            "ON" if rectangle else "OFF",
                                                                                                            "ON" if trace else "OFF")
    cv2.putText(frame,
                message,
                (20, 20),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                1)

    cv2.imshow("frame", frame)
    key=cv2.waitKey(70)&0xFF
    if key==ord('r'):
        rectangle=not rectangle
    if key==ord('t'):
        trace=not trace
    if key==ord('q'):
        quit()
    id_frame+=1
