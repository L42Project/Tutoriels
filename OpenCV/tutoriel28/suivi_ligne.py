import cv2
import numpy as np

lo=np.array([60, 30, 30])
hi=np.array([100, 255, 255])
cap=cv2.VideoCapture(0)

taille_objet=15
nbr_point=100
tab_point=np.full((nbr_point, 2), -1, dtype=np.int32)

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

    old_x, old_y=(-1, -1)
    for i in range(nbr_point):
        if tab_point[nbr_point-i-1, 0]!=-1:
            if old_x!=-1:
                cv2.line(frame, (old_x, old_y), (tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1]), (0, 255-2*(nbr_point-i-1), 0), 10)
                #cv2.line(frame, (old_x, old_y), (tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1]), (0, 255, 0), 10)
            old_x, old_y=(tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1])

    cv2.imshow('Camera', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
