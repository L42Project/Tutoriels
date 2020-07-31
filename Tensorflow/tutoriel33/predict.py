import random
import tensorflow as tf
import csv
import numpy as np
import cv2
import model

fichier='ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
dir_images='ISIC2018_Task3_Training_Input/'

labels=['Melanoma',
        'Melanocytic nevus',
        'Basal cell carcinoma',
        'Actinic keratosis',
        'Benign keratosis',
        'Dermatofibroma',
        'Vascular lesion']

tab_images=[]
tab_labels=[]

def rotateImage(image, angle):
    image_center=tuple(np.array(image.shape[1::-1])/2)
    rot_mat=cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result=cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def bruit(image):
    h, w, c=image.shape
    n=np.random.randn(h, w, c)*random.randint(5, 30)
    return np.clip(image+n, 0, 255).astype(np.uint8)

with open(fichier, newline='') as csvfile:
    lignes=csv.reader(csvfile, delimiter=',')
    next(lignes, None)
    for ligne in lignes:
        label=np.array(ligne[1:], dtype=np.float32)        
        img=cv2.imread(dir_images+ligne[0]+'.jpg')
        img=cv2.resize(img, (100, 75))
        if img is None:
            print("XXX")
            quit()
        tab_labels.append(label)
        tab_images.append(img)

        if label[1]:
            continue

        flag=0
        for angle in range(0, 360, 30):
            img_r=rotateImage(img, angle)
            
            if label[2] or label[3] or label[5] or label[6]:
                tab_labels.append(label)
                i=cv2.flip(img_r, 0)
                tab_images.append(i)

            if not flag%3 and (label[0] or label[4]):
                tab_labels.append(label)
                i=cv2.flip(img_r, 0)
                tab_images.append(i)
            flag+=1
                
            if label[2] or label[3] or label[5] or label[6]:
                tab_labels.append(label)
                i=cv2.flip(img_r, 1)
                tab_images.append(i)

            if label[5] or label[6]:
                tab_labels.append(label)
                i=cv2.flip(img_r, -1)
                tab_images.append(i)

tab_labels=np.array(tab_labels, dtype=np.float32)        
tab_images=np.array(tab_images, dtype=np.float32)/255

indices=np.random.permutation(len(tab_labels))
tab_labels=tab_labels[indices]
tab_images=tab_images[indices]

print("SOMME", np.sum(tab_labels, axis=0))

model=tf.keras.models.load_model('my_model/')

for i in range(len(tab_images)):
    cv2.imshow("image", tab_images[i])
    prediction=model.predict(np.array([tab_images[i]], dtype=np.float32))
    print("Bonne reponse:{}, Reponse du réseau:{}".format(labels[np.argmax(tab_labels[i])], labels[np.argmax(prediction[0])]))
    key=cv2.waitKey()&0xFF
    if key==ord('q'):
        break

cv2.destroyAllWindows()
