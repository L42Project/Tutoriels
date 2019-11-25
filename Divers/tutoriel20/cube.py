import cv2
import numpy as np
import dlib
import math

cap=cv2.VideoCapture(0)
#cap=cv2.VideoCapture("debat.webm")

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def tr(c, o, coeff):
    return(int((c-o)*coeff)+o)

def cube(image, pt1, pt2, a1, a2, a3):
    color=(0, 255, 0)
    epaisseur=2
    offset=1.6
    offset2=2

    d_eyes=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(45).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(45).y, 2))

    ox1=int((-(pt2.y-pt1.y)+pt2.x-pt1.x)/2)+pt1.x
    oy1=int(((pt2.x-pt1.x+pt2.y)-pt1.y)/2)+pt1.y

    cv2.line(image,
        (tr(pt1.x, ox1, offset), tr(pt1.y, oy1, offset)),
        (tr(pt2.x, ox1, offset), tr(pt2.y, oy1, offset)),
        color, epaisseur)
    cv2.line(image,
        (tr(pt2.x, ox1, offset), tr(pt2.y, oy1, offset)),
        (tr(-(pt2.y-pt1.y)+pt2.x, ox1, offset), tr(pt2.x-pt1.x+pt2.y, oy1, offset)),
        color, epaisseur)
    cv2.line(image,
        (tr(-(pt2.y-pt1.y)+pt2.x, ox1, offset), tr(pt2.x-pt1.x+pt2.y, oy1, offset)),
        (tr(-(pt2.y-pt1.y)+pt1.x, ox1, offset), tr(pt2.x-pt1.x+pt1.y, oy1, offset)),
        color, epaisseur)
    cv2.line(image,
        (tr(-(pt2.y-pt1.y)+pt1.x, ox1, offset), tr(pt2.x-pt1.x+pt1.y, oy1, offset)),
        (tr(pt1.x, ox1, offset), tr(pt1.y, oy1, offset)),
        color, epaisseur)

    ox2=int((-(pt2.y-pt1.y)+pt2.x-pt1.x)/2)+pt1.x+int(a2)
    oy2=int(((pt2.x-pt1.x+pt2.y)-pt1.y)/2)+pt1.y+int(a3)

    cv2.line(image,
        (tr(pt1.x+a2, ox2, offset2), tr(pt1.y+a3, oy2, offset2)),
        (tr(pt2.x+a2, ox2, offset2), tr(pt2.y+a3, oy2, offset2)),
        color, epaisseur)
    cv2.line(image,
        (tr(pt2.x+a2, ox2, offset2), tr(pt2.y+a3, oy2, offset2)),
        (tr(-(pt2.y-pt1.y)+pt2.x+a2, ox2, offset2), tr(pt2.x-pt1.x+pt2.y+a3, oy2, offset2)),
        color, epaisseur)
    cv2.line(image,
        (tr(-(pt2.y-pt1.y)+pt2.x+a2, ox2, offset2), tr(pt2.x-pt1.x+pt2.y+a3, oy2, offset2)),
        (tr(-(pt2.y-pt1.y)+pt1.x+a2, ox2, offset2), tr(pt2.x-pt1.x+pt1.y+a3, oy2, offset2)),
        color, epaisseur)
    cv2.line(image,
        (tr(-(pt2.y-pt1.y)+pt1.x+a2, ox2, offset2), tr(pt2.x-pt1.x+pt1.y+a3, oy2, offset2)),
        (tr(pt1.x+a2, ox2, offset2), tr(pt1.y+a3, oy2, offset2)),
        color, epaisseur)

    cv2.line(image,
        (tr(pt1.x, ox1, offset), tr(pt1.y, oy1, offset)),
        (tr(pt1.x+a2, ox2, offset2), tr(pt1.y+a3, oy2, offset2)),
        color, epaisseur)
    cv2.line(image,
        (tr(pt2.x, ox1, offset), tr(pt2.y, oy1, offset)),
        (tr(pt2.x+a2, ox2, offset2), tr(pt2.y+a3, oy2, offset2)),
        color, epaisseur)
    cv2.line(image,
        (tr(-(pt2.y-pt1.y)+pt2.x, ox1, offset), tr(pt2.x-pt1.x+pt2.y, oy1, offset)),
        (tr(-(pt2.y-pt1.y)+pt2.x+a2, ox2, offset2), tr(pt2.x-pt1.x+pt2.y+a3, oy2, offset2)),
        color, epaisseur)
    cv2.line(image,
        (tr(-(pt2.y-pt1.y)+pt1.x, ox1, offset), tr(pt2.x-pt1.x+pt1.y, oy1, offset)),
        (tr(-(pt2.y-pt1.y)+pt1.x+a2, ox2, offset2), tr(pt2.x-pt1.x+pt1.y+a3, oy2, offset2)),
        color, epaisseur)

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

        landmarks=predictor(gray, face)

        d_eyes=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(45).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(45).y, 2))
        d1=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(30).y, 2))
        d2=math.sqrt(math.pow(landmarks.part(45).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(45).y-landmarks.part(30).y, 2))
        coeff=d1+d2

        a1=int(250*(landmarks.part(36).y-landmarks.part(45).y)/coeff)
        a2=int(250*(d1-d2)/coeff)
        cosb=min((math.pow(d2, 2)-math.pow(d1, 2)+math.pow(d_eyes, 2))/(2*d2*d_eyes), 1)
        a3=int(250*(d2*math.sin(math.acos(cosb))-coeff/4)/coeff)

        cube(frame, landmarks.part(36), landmarks.part(45), a1, a2, a3)
    cv2.imshow("Frame", frame)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
1
