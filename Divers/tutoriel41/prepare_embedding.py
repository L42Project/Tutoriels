import face_recognition
import numpy as np
import os
import glob

dir_identites="Identites_1photo/"

face_encodings=[]
face_names=[]

id=1
for dir_identite in os.listdir(dir_identites):
    print("ID", id)
    id+=1

    fichiers=[]
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        fichiers.extend(glob.glob(dir_identites+"/"+dir_identite+"/"+ext))
    if len(fichiers)==0:
        print("Repertoire vide", dir_identite)
        continue
    for fichier in fichiers:
        image=face_recognition.load_image_file(fichier)
        embedding=face_recognition.face_encodings(image)[0]
        face_encodings.append(embedding)
        face_names.append(dir_identite.replace('_', ' '))
            
np.save("face_encodings", np.array(face_encodings))
np.save("face_names", np.array(face_names))



