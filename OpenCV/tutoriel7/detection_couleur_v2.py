import cv2
import numpy as np

def souris(event, x, y, flags, param):
    global lo, hi, color
    if event==cv2.EVENT_LBUTTONDBLCLK:
        color=image[y, x][0]
    if event==cv2.EVENT_MOUSEWHEEL:
        if flags<0:
            if color>5:
                color-=1
        else:
            if color<250:
                color+=1
    lo[0]=color-5
    hi[0]=color+5

color=100
lo=np.array([color-5, 100, 50])
hi=np.array([color+5, 255,255])
color_info=(0, 0, 255)
cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image=cv2.blur(image, (5, 5))
    mask=cv2.inRange(image, lo, hi)
    mask=cv2.erode(mask, None, iterations=1)
    mask=cv2.dilate(mask, None, iterations=1)
    image2=cv2.bitwise_and(frame, frame, mask= mask)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c=max(elements, key=cv2.contourArea)
        ((x, y), radius)=cv2.minEnclosingCircle(c)
        if radius>30:
            cv2.circle(frame, (int(x), int(y)), 5, color_info, 10)
            cv2.line(frame, (int(x), int(y)), (int(x)+150, int(y)), color_info, 2)
            cv2.putText(frame, "Objet !!!", (int(x)+10, int(y) -10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
    cv2.putText(frame, "Couleur: {:d}".format(color), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
    cv2.imshow('Camera', frame)
    cv2.imshow('image2', image2)
    cv2.imshow('Mask', mask)
    cv2.setMouseCallback('Camera', souris)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
