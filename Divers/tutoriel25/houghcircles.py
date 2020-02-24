import tensorflow as tf
import cv2
import os
import numpy as np
import random
import common

param1=30
param2=55
dp=1.0

cap=cv2.VideoCapture(1)

while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, 20, param1=param1, param2=param2, minRadius=10, maxRadius=50)
    if circles is not None:
        circles=np.around(circles)
        for i in circles[0, :]:
            if i[2]!=0:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 4)
    cv2.putText(frame, "[i|k]dp: {:4.2f}  [o|l]param1: {:d}  [p|m]param2: {:d}".format(dp, param1, param2), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    cv2.imshow("Video", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        quit()
    if key==ord('i'):
        dp=min(10, dp+0.1)
    if key==ord('k'):
        dp=max(0.1, dp-0.1)
    if key==ord('o'):
        param1=min(255, param1+1)
    if key==ord('l'):
        param1=max(1, param1-1)
    if key==ord('p'):
        param2=min(255, param2+1)
    if key==ord('m'):
        param2=max(1, param2-1)
cv2.destroyAllWindows()
