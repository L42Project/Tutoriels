import cv2
import os
import numpy as np
import pickle
import common as c

image_dir="./photos/"
current_id=0
label_ids={}
x_train=[]
y_labels=[]

for root, dirs, files in os.walk(image_dir):
    if len(files):
        label=root.split("/")[-1]
        for file in files:
            if file.endswith("png"):
                path=os.path.join(root, file)
                if not label in label_ids:
                    label_ids[label]=current_id
                    current_id+=1
                id_=label_ids[label]
                image=cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (c.min_size, c.min_size))
                fm=cv2.Laplacian(image, cv2.CV_64F).var()
                if fm<250:
                    print("Photo exclue:", path, fm)
                else:
                    x_train.append(image)
                    y_labels.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

x_train=np.array(x_train)
y_labels=np.array(y_labels)
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train, y_labels)
recognizer.save("trainner.yml")
