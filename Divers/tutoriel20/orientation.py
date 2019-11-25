import cv2
import numpy as np
import dlib
import math

cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces=detector(gray)
    if faces is not None:
        i=np.zeros(shape=(frame.shape), dtype=np.uint8)
    for face in faces:
        landmarks=predictor(gray, face)

        d_eyes=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(45).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(45).y, 2))
        d1=math.sqrt(math.pow(landmarks.part(36).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(36).y-landmarks.part(30).y, 2))
        d2=math.sqrt(math.pow(landmarks.part(45).x-landmarks.part(30).x, 2)+math.pow(landmarks.part(45).y-landmarks.part(30).y, 2))
        coeff=d1+d2

        a1=int(250*(landmarks.part(36).y-landmarks.part(45).y)/coeff)
        a2=int(250*(d1-d2)/coeff)
        cosb=min((math.pow(d2, 2)-math.pow(d1, 2)+math.pow(d_eyes, 2))/(2*d2*d_eyes), 1)
        a3=int(250*(d2*math.sin(math.acos(cosb))-coeff/4)/coeff)

        for n in range(0, 68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            if n==30 or n==36 or n==45:
                cv2.circle(i, (x, y), 3, (255, 255, 0), -1)
            else:
                cv2.circle(i, (x, y), 3, (255, 0, 0), -1)
        print("{:+05d} {:+05d} {:+05d}".format(a1, a2, a3))
        flag=1
        txt="Laurent regarde "
        if a2<-40:
            txt+="a droite "
            flag=0
        if a2>40:
            txt+="a gauche "
            flag=0
        if a3<-10:
            txt+="en haut "
            flag=0
        if a3>10:
            txt+="en bas "
            flag=0
        if flag:
            txt+="la camera "
        if a1<-40:
            txt+="et incline la tete a gauche "
        if a1>40:
            txt+="et incline la tete a droite "
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
