import cv2
import numpy as np
import dlib

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
