import cv2
import numpy as np
import tensorflow as tf
import sudoku_solver as ss
from time import sleep
import operator

methode=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
v1=9

cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (5, 5), 0)
    #ret, thresh=cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    thresh=cv2.adaptiveThreshold(gray, 255, methode, cv2.THRESH_BINARY_INV, v1, 2)
    cv2.imshow("thresh", thresh)
    txt="ADAPTIVE_THRESH_MEAN_C" if methode==cv2.ADAPTIVE_THRESH_MEAN_C else "ADAPTIVE_THRESH_GAUSSIAN_C"
    cv2.putText(frame, "[u|j]v1: {:2d}  [o]methode: {}".format(v1, txt), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)
    cv2.imshow("frame", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('p'):
        v1=min(21, v1+2)
    if key==ord('m'):
        v1=max(3, v1-2)
    if key==ord('o'):
        if methode==cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
            methode=cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            methode=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
cap.release()
cv2.destroyAllWindows()
