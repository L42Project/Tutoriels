import cv2
import numpy as np

def moyenne_image(video, nbr):
    cap=cv2.VideoCapture(video)
    tab_image=[]
    for f in range(nbr):
        ret, frame=cap.read()
        if ret is False:
            break
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tab_image.append(image)
    tab_image=np.array(tab_image)
    return np.mean(tab_image, axis=0)

def calcul_mask(image, fond, seuil):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width=image.shape
    mask=np.zeros([height, width], np.uint8)
    image=image.astype(np.int32)
    for y in range(height):
        for x in range(width):
            if abs(fond[y][x]-image[y][x])>seuil:
                mask[y][x]=255
    kernel=np.ones((5, 5), np.uint8)
    mask=cv2.erode(mask, kernel, iterations=1)
    mask=cv2.dilate(mask, kernel, iterations=3)
    return mask
