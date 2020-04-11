import cv2
import numpy as np

lo=np.array([60, 30, 30])
hi=np.array([100, 255, 255])
cap=cv2.VideoCapture(0)

width=cap.get(3)
height=cap.get(4)

nbr_point=100
tab_point=np.full((nbr_point, 2), -1, dtype=np.int32)

# mode 0 : point
# mode 1 : ligne
mode=1
degrade=1
taille_objet=15

def dessine_point(tab_point):
    for i in range(len(tab_point)):
        if tab_point[nbr_point-i-1, 0]!=-1:
            if degrade:
                couleur=(0, 255-2*(nbr_point-i-1), 0)
            else:
                couleur=(0, 255, 0)
            cv2.circle(frame, (tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1]), 5, couleur, 10)

def dessine_ligne(tab_point):
    old_x, old_y=(-1, -1)
    for i in range(nbr_point):
        if tab_point[nbr_point-i-1, 0]!=-1:
            if old_x!=-1:
                if degrade:
                    couleur=(0, 255-2*(nbr_point-i-1), 0)
                else:
                    couleur=(0, 255, 0)
                cv2.line(frame, (old_x, old_y), (tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1]), couleur, 10)
        old_x, old_y=(tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1])

while True:
    ret, frame=cap.read()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image=cv2.blur(image, (5, 5))
    mask=cv2.inRange(image, lo, hi)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=4)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    tab_point=np.roll(tab_point, 1, axis=0)
    tab_point[0]=[-1, -1]
    if len(elements) > 0:
        c=max(elements, key=cv2.contourArea)
        ((x, y), rayon)=cv2.minEnclosingCircle(c)
        if rayon>taille_objet:
            tab_point[0]=[int(x), int(y)]

    if mode:
        dessine_ligne(tab_point)
    else:
        dessine_point(tab_point)

    cv2.rectangle(frame, (0, 0), (int(width), 30), (100, 100, 100), cv2.FILLED)
    cv2.putText(frame, "Mode[m]: {:d}   Degrade[p]: {:d}".format(mode, degrade), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Camera', frame)
    cv2.imshow('Mask', mask)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('m'):
        mode=not mode
    if key==ord('p'):
        degrade=not degrade
cap.release()
cv2.destroyAllWindows()
