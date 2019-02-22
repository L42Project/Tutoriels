import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
cap=cv2.VideoCapture('test.mp4')

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('video', frame)
    if cv2.waitKey(50)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
