import os
import sys
import time
import numpy as np
import cv2

cap=cv2.VideoCapture("chien.mp4") # Mettre votre video ou webcam!

kernel_blur=3
seuil=15
surface=6000
ret, originale=cap.read()
if ret is False:
    quit()
originale=cv2.cvtColor(originale, cv2.COLOR_BGR2GRAY)
originale=cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)
kernel_dilate=np.ones((5, 5), np.uint8)
alarme=0
intrus=0
while True:
    ret, frame=cap.read()
    if ret is False:
        quit()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur=cv2.GaussianBlur(gray, (kernel_blur, kernel_blur), 0)
    mask=cv2.absdiff(originale, gray_blur)
    mask=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)[1]
    mask=cv2.dilate(mask, kernel_dilate, iterations=3)
    contours, nada=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_contour=frame.copy()
    for c in contours:
        if cv2.contourArea(c)<surface:
            continue
        cv2.drawContours(frame_contour, [c], 0, (0, 255, 0), 2)
        x, y, w, h=cv2.boundingRect(c)
        alarme=1
        intrus=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if intrus:
            cv2.putText(frame, "INTRUS", (x, y-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    originale=gray_blur
    if alarme:
        cv2.putText(frame, "ALARME", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    cv2.putText(frame, "[o|l]seuil: {:d}  [p|m]blur: {:d}  [i|k]surface: {:d}".format(seuil, kernel_blur, surface), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
    cv2.imshow("frame", frame)
    cv2.imshow("Mask", mask)
    intrus=0
    key=cv2.waitKey(30)&0xFF
    if key==ord('q'):
        break
    if key==ord('p'):
        kernel_blur=min(43, kernel_blur+2)
    if key==ord('m'):
        kernel_blur=max(1, kernel_blur-2)
    if key==ord('o'):
        seuil=min(255, seuil+1)
    if key==ord('l'):
        seuil=max(1, seuil-1)
    if key==ord('i'):
        surface+=1000
    if key==ord('k'):
        surface=max(1000, surface-1000)
    if key==ord('a'):
        alarme=0
cap.release()
cv2.destroyAllWindows()
