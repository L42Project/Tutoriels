import cv2
import numpy as np
import random

def rotateImage(image, angle):
    image_center=tuple(np.array(image.shape[1::-1])/2)
    rot_mat=cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result=cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def bruit(image):
    h, w, c=image.shape
    n=np.random.randn(h, w, c)*random.randint(5, 30)
    return np.clip(image+n, 0, 255).astype(np.uint8)

def change_gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

def color(image, alpha=20):
    n=[random.randint(-alpha, alpha), random.randint(-alpha, alpha),random.randint(-alpha, alpha)]
    return np.clip(image+n, 0, 255).astype(np.uint8)

def random_change(image):
    if np.random.randint(2):
        img=change_gamma(image, random.uniform(0.8, 1.2), np.random.randint(100)-50)
    if np.random.randint(2):
        img=bruit(image)
    if np.random.randint(2):
        img=color(image)
    return image
