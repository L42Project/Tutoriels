import cv2
import operator
import os
import common as c

video="Plan_9_from_Outer_Space_1959_512kb.mp4"
cascade="./haarcascade_frontalface_alt2.xml"
img_non_classees='non-classees'

if not os.path.exists(video):
    print("Le fichier video n'existe pas", video)
    quit()

if not os.path.exists(cascade):
    print("Le fichier cascade n'existe pas", cascade)
    quit()

if not os.path.isdir(img_non_classees):
    os.mkdir(img_non_classees)

face_cascade=cv2.CascadeClassifier(cascade)
cap=cv2.VideoCapture(video)

id=0
while True:
    ret, frame=cap.read()
    if ret is False:
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(c.min_size, c.min_size))
    for x, y, w, h in face:
        cv2.imwrite("{}/p-{:d}.png".format(img_non_classees, id), frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id+=1
    cv2.imshow('video', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100):
            ret, frame=cap.read()

cap.release()
cv2.destroyAllWindows()
