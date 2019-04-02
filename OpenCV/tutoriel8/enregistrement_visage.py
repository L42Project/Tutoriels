import cv2
import operator
import common as c

face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
cap=cv2.VideoCapture("Plan 9 from Outer Space Charles Burg, J. Edward Reynolds, Hu.mp4")

id=0
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(c.min_size, c.min_size))
    for x, y, w, h in face:
        cv2.imwrite("non-classees/p-{:d}.png".format(id), frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id+=1
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100):
            ret, frame=cap.read()

    cv2.imshow('video', frame)

cap.release()
cv2.destroyAllWindows()
