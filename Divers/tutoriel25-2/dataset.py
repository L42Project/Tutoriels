import numpy as np
import cv2
from multiprocessing import Pool
import multiprocessing
import random

def bruit(image_orig):
    h, w, c=image_orig.shape
    n=np.random.randn(h, w, c)*random.randint(5, 30)
    return np.clip(image_orig+n, 0, 255).astype(np.uint8)

def change_gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)
    
def modif_img(img):
    h, w, c=img.shape

    r_color=[np.random.randint(255), np.random.randint(255), np.random.randint(255)]
    img=np.where(img==[142, 142, 142], r_color, img).astype(np.uint8)

    if np.random.randint(3):
        k_max=3
        kernel_blur=np.random.randint(k_max)*2+1
        img=cv2.GaussianBlur(img, (kernel_blur, kernel_blur), 0)

    M=cv2.getRotationMatrix2D((int(w/2), int(h/2)), random.randint(-10, 10), 1)
    img=cv2.warpAffine(img, M, (w, h))
        
    if np.random.randint(2):
        a=int(max(w, h)/5)+1
        pts1=np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2=np.float32([[0+random.randint(-a, a), 0+random.randint(-a, a)], [w-random.randint(-a, a), 0+random.randint(-a, a)], [0+random.randint(-a, a), h-random.randint(-a, a)], [w-random.randint(-a, a), h-random.randint(-a, a)]])        
        M=cv2.getPerspectiveTransform(pts1,pts2)
        img=cv2.warpPerspective(img, M, (w, h))
        
    if np.random.randint(2):
        r=random.randint(0, 5)
        h2=int(h*0.9)
        w2=int(w*0.9)
        if r==0:
            img=img[0:w2, 0:h2]
        elif r==1:
            img=img[w-w2:w, 0:h2]
        elif r==2:
            img=img[0:w2, h-h2:h]
        elif r==3:
            img=img[w-w2:w, h-h2:h]
        img=cv2.resize(img, (h, w))

    if np.random.randint(2):
        r=random.randint(1, int(max(w, h)*0.15))
        img=img[r:w-r, r:h-r]
        img=cv2.resize(img, (h, w))

    if not np.random.randint(4):
        t=np.empty((h, w, c) , dtype=np.float32)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    t[i][j][k]=(i/h)
        M=cv2.getRotationMatrix2D((int(w/2), int(h/2)), np.random.randint(4)*90, 1)
        t=cv2.warpAffine(t, M, (w, h))
        img=(cv2.multiply((img/255).astype(np.float32), t)*255).astype(np.uint8)

    img=change_gamma(img, random.uniform(0.6, 1.0), -np.random.randint(50))

    if not np.random.randint(4):
        p=(15+np.random.randint(10))/100
        img=(img*p+50*(1-p)).astype(np.uint8)+np.random.randint(100)

    img=bruit(img)
        
    return img

def create_lot_img(image, nbr, nbr_thread=None):
    if nbr_thread is None:
        nbr_thread=multiprocessing.cpu_count()
    lot_original=np.repeat([image], nbr, axis=0)
    with Pool(nbr_thread) as p:
        lot_result=p.map(modif_img, lot_original)
        p.close()
    return lot_result
