import tensorflow as tf
from tensorflow.keras import layers, models
import csv
import random
import cv2
import numpy as np
import math
import csv
import random
import config

def rotateImage(image, angle):
    image_center=tuple(np.array(image.shape[1::-1])/2)
    rot_mat=cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result=cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def complete_dataset(image, image_ellipse, tab_images, tab_labels):
    contours, hierarchy=cv2.findContours(image_ellipse[:, :, 0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) is not 2:
        return 1
    else:
        if len(contours[0])<6 or len(contours[1])<6:
            return 1
        (x1, y1), (ma1, MA1), a1=cv2.fitEllipse(contours[0])
        (x2, y2), (ma2, MA2), a2=cv2.fitEllipse(contours[1])
        x=(x1+x2)/2
        y=(y1+y2)/2
        ma=(ma1+ma2)/2
        MA=(MA1+MA2)/2
        a=(a1+a2)/2
        tab_images.append(image[:, :, 0])
        tab_labels.append([x/config.norm, y/config.norm, MA/config.norm, ma/config.norm, a/180])
    return 0
        
def prepare_data(fichier):
  with open(fichier, newline='') as csvfile:
    lignes=csv.reader(csvfile, delimiter=',')
    next(lignes)
    tab_images=[]
    tab_labels=[]
    nbr=0
    for ligne in lignes:
        image_orig=cv2.imread(config.dir_images+ligne[0])
        if image_orig is None:
            print("Fichier absent", config.dir_images+ligne[0])
            continue

        f_ellipse=ligne[0].split('.')[0]+"_Annotation.png"
        image_ellipse_orig=cv2.imread(config.dir_images+f_ellipse)
        if image_ellipse_orig is None:
            print("Fichier absent", config.dir_images+f_ellipse)
            continue
    
        for angle in range(0, 360, 30):
            
            if np.random.randint(2)==0:
                h, w, c=image_orig.shape
                H=int(h*1.4)
                W=int(w*1.4)
                h_shift=np.random.randint(H-h)
                w_shift=np.random.randint(W-w)
                
                i=np.zeros(shape=(H, W, c), dtype=np.uint8)
                i[h_shift:h_shift+h, w_shift:w_shift+w, :]=image_orig
                image_orig2=i
                
                i=np.zeros(shape=(H, W, c), dtype=np.uint8)
                i[h_shift:h_shift+h, w_shift:w_shift+w, :]=image_ellipse_orig
                image_ellipse_orig2=i
            else:
                image_orig2=image_orig
                image_ellipse_orig2=image_ellipse_orig

            image=cv2.resize(image_orig2, (config.largeur, config.hauteur), interpolation=cv2.INTER_AREA)
            image_ellipse=cv2.resize(image_ellipse_orig2, (config.largeur, config.hauteur), interpolation=cv2.INTER_AREA)
            img_r=rotateImage(image, angle)

            #if np.random.randint(3)==0:
            #    kernel_blur=np.random.randint(2)*2+1
            #    img_r=cv2.GaussianBlur(img_r, (kernel_blur, kernel_blur), 0)
        
            bruit=np.random.randn(config.hauteur, config.largeur, 3)*random.randint(1, 50)
            img_r=np.clip(img_r+bruit, 0, 255).astype(np.uint8)
            
            img_ellipse=rotateImage(image_ellipse, angle)
            nbr+=complete_dataset(img_r, img_ellipse, tab_images, tab_labels)
            
            img_f=cv2.flip(img_r, 0)
            img_ellipse_f=cv2.flip(img_ellipse, 0)
            nbr+=complete_dataset(img_f, img_ellipse_f, tab_images, tab_labels)
            
            img_f=cv2.flip(img_r, 1)
            img_ellipse_f=cv2.flip(img_ellipse, 1)
            nbr+=complete_dataset(img_f, img_ellipse_f, tab_images, tab_labels)
            
            img_f=cv2.flip(img_r, -1)
            img_ellipse_f=cv2.flip(img_ellipse, -1)
            nbr+=complete_dataset(img_f, img_ellipse_f, tab_images, tab_labels)
            
    print("Image(s) rejetée(s):", nbr)
    print("Nombre d'images:", len(tab_images))
    return tab_images, tab_labels  
      
