import numpy as np
import cv2

lo=np.array([80, 50, 50])
hi=np.array([100, 255, 255])

def detect_inrange(image, surface):
    points=[]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=cv2.blur(image, (5, 5))
    mask=cv2.inRange(image, lo, hi)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=2)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
    for element in elements:
        if cv2.contourArea(element)>surface:
            ((x, y), rayon)=cv2.minEnclosingCircle(element)
            points.append(np.array([int(x), int(y)]))
        else:
            break

    return points, mask

def detect_visage(image):
    face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
    points=[]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    for x, y, w, h in face:
        points.append(np.array([int(x+w/2), int(y+h/2)]))

    return points, None
