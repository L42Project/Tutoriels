import cv2
import numpy as np
import dlib
import math

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        landmarks=predictor(gray, face)
        i=np.zeros(shape=(frame.shape), dtype=np.uint8)
        for n in range(0, 68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            if n==30 or n==36 or n==45:
                cv2.circle(i, (x, y), 3, (255, 255, 0), -1)
            else:
                cv2.circle(i, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("i", i)
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
