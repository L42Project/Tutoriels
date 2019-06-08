import cv2
import os

dir='dataA/'

for file in os.listdir(dir+'CameraRGB/'):
    img=cv2.imread(dir+'CameraRGB/'+file)
    cv2.imshow("image", img)
    mask=cv2.imread(dir+'CameraSeg/'+file)
    cv2.imshow("mask", mask*25)
    key=cv2.waitKey()&0xFF
    if key==ord('q'):
        quit()
