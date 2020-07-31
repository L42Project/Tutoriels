import cv2
import os

dir='./predictions/'
dir2='./predictions2/'

for fichier in os.listdir(dir):
    img=cv2.imread(dir+fichier)
    img[img<50]=0
    img[img>200]=255
    cv2.imwrite(dir2+fichier, img)
    
