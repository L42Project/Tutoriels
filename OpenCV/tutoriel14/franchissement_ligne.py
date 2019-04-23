import cv2
import numpy as np
import time

ymin=320
ymax=321
xmin1=80
xmax1=170
xmin2=410
xmax2=500

def point(capteur):
    s1=len(capteur)-1
    s2=len(capteur)-1
    for i in range(len(capteur)):
        if capteur[i]!=0:
            s1=i
            break
    if s1!=len(capteur)-1:
        for i in range(len(capteur)-1, s1-1, -1):
            if capteur[i]!=0:
                s2=i
                break
        return int((s1+s2)/2)
    return -1

s1_old=0
s2_old=0
s1=0
s2=0
s1_time=0
s2_time=0
th1=75
th2=150
stop=0
k=1
cap=cv2.VideoCapture("autoroute.mp4")
while True:
    if not stop:
        ret, frame=cap.read()
        if ret is False:
            quit()
        image=frame.copy()

        gray1=cv2.cvtColor(image[ymin:ymax, xmin1:xmax1], cv2.COLOR_BGR2GRAY)
        if k!=1:
            gray1=cv2.blur(gray1, (k, k))
        capteur1=cv2.Canny(gray1, th1, th2)

        gray2=cv2.cvtColor(image[ymin:ymax, xmin2:xmax2], cv2.COLOR_BGR2GRAY)
        if k!=1:
            gray2=cv2.blur(gray2, (k, k))
        capteur2=cv2.Canny(gray2, th1, th2)

        cv2.rectangle(image, (xmin1, ymin), (xmax1, ymax), (0, 0, 255), 1)
        cv2.rectangle(image, (xmin2, ymin), (xmax2, ymax), (0, 0, 255), 1)

        s1=point(capteur1[0])
        if s1!=-1:
            cv2.circle(image, (s1+xmin1, ymin), 3, (0, 255, 0), 3)
            s1_old=s1
            s1_time=time.time()
        else:
            if time.time()-s1_time<1:
                cv2.circle(image, (s1_old+xmin1, ymin), 3, (100, 255, 255), 3)
                s1=s1_old
            else:
                s1=-1

        s2=point(capteur2[0])
        if s2!=-1:
            cv2.circle(image, (s2+xmin2, ymin), 3, (0, 255, 0), 3)
            s2_old=s2
            s2_time=time.time()
        else:
            if time.time()-s2_time<1:
                cv2.circle(image, (s2_old+xmin2, ymin), 3, (100, 255, 255), 3)
                s2=s2_old
            else:
                s2=-1

        if s1!=-1 and s2!=-1:
            s2_=abs(xmax2-xmin2-s2)
            if abs(s2_-s1)>20:
                c=(0, max(0, 255-10*int(abs(s1-s2_)/2)), min(255, 10*int(abs(s1-s2_)/2)))
                cv2.circle(image, (int((xmax2-xmin1)/2)+xmin1, ymax-25), 5, c, 7)
                cv2.arrowedLine(image, (int((xmax2-xmin1)/2)+xmin1, ymax-25), (int((xmax2-xmin1)/2)+xmin1+2*int((s1-s2_)/2), ymax-25), c, 3, tipLength=0.4)
            else:
                cv2.putText(image, "OK", (int((xmax2-xmin1)/2)+xmin1-15, ymax-16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    cv2.putText(image, "[u|j]th1: {:d}  [i|k]th2: {:d}  [y|h]blur: {:d}".format(th1, th2, k), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)
    cv2.putText(image, "[a]>>  [s]stop  [q]quit", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)
    cv2.imshow("image", image)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if k!=1:
        gray=cv2.blur(gray, (k, k))
    cv2.imshow("blur", gray)
    gray_canny=cv2.Canny(gray, th1, th2)
    cv2.imshow("canny", gray_canny)

    if not stop:
        key=cv2.waitKey(20)&0xFF
    else:
        key=cv2.waitKey()
        image=frame.copy()

    if key==ord('q'):
        break
    if key==ord('m'):
        ymin+=1
        ymax+=1
    if key==ord('p'):
        ymin-=1
        ymax-=1
    if key==ord('o'):
        xmin1+=1
        xmax1+=1
        xmin2+=1
        xmax2+=1
    if key==ord('l'):
        xmin1-=1
        xmax1-=1
        xmin2-=1
        xmax2-=1
    if key==ord('y'):
        k=min(255, k+2)
    if key==ord('h'):
        k=max(1, k-2)
    if key==ord('u'):
        th1=min(255, th1+1)
    if key==ord('j'):
        th1=max(0, th1-1)
    if key==ord('i'):
        th2=min(255, th2+1)
    if key==ord('k'):
        th2=max(0, th2-1)
    if key==ord('s'):
        stop=not stop
    if key==ord('a'):
        for cpt in range(200):
            ret, frame=cap.read()
            image=frame.copy()

cap.release()
cv2.destroyAllWindows()
