import cv2
import numpy as np

lo=np.array([95, 100, 50])
hi=np.array([105, 255, 255])
color_infos=(0, 255, 255)
cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(image, lo, hi)
    image=cv2.blur(image, (7, 7))
    mask=cv2.erode(mask, None, iterations=4)
    mask=cv2.dilate(mask, None, iterations=4)
    image2=cv2.bitwise_and(frame, frame, mask=mask)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c=max(elements, key=cv2.contourArea)
        ((x, y), rayon)=cv2.minEnclosingCircle(c)
        if rayon>30:
            cv2.circle(image2, (int(x), int(y)), int(rayon), color_infos, 2)
            cv2.circle(frame, (int(x), int(y)), 5, color_infos, 10)
            cv2.line(frame, (int(x), int(y)), (int(x)+150, int(y)), color_infos, 2)
            cv2.putText(frame, "Objet !!!", (int(x)+10, int(y) -10), cv2.FONT_HERSHEY_DUPLEX, 1, color_infos, 1, cv2.LINE_AA)
    cv2.imshow('Camera', frame)
    cv2.imshow('image2', image2)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
