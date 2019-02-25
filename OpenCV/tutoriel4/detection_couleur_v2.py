import cv2
import numpy as np

def souris(event, x, y, flags, param):
    global lo, hi, color
    if event == cv2.EVENT_LBUTTONDBLCLK:
        color=image[y, x][0]
        lo[0]=color-5
        hi[0]=color+5

color=100
lo=np.array([color-5, 100, 50])
hi=np.array([color+5, 255, 255])
color_infos=(0, 255, 255)
cap=cv2.VideoCapture(1)
kernel = np.ones((5,5),np.uint8)

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(image, lo, hi)
    mask=cv2.erode(mask, kernel, iterations=3)
    mask=cv2.dilate(mask, kernel, iterations=3)
    image2=cv2.bitwise_and(frame, frame, mask= mask)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c=max(elements, key=cv2.contourArea)
        ((x, y), rayon)=cv2.minEnclosingCircle(c)
        if rayon>30:
            cv2.circle(image2, (int(x), int(y)), int(rayon), color_infos, 2)
            cv2.circle(frame, (int(x), int(y)), 5, color_infos, 10)
            cv2.line(frame, (int(x), int(y)), (int(x)+150, int(y)), color_infos, 2)
            cv2.putText(frame, "Objet !!!", (int(x)+10, int(y) -10), cv2.FONT_HERSHEY_DUPLEX, 1, color_infos, 1, cv2.LINE_AA)
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:06.2f} - couleur: {:03d}".format(fps, color), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_infos, 2)
    cv2.imshow('Camera', frame)
    cv2.imshow('image2', image2)
    cv2.imshow('Mask', mask)
    cv2.setMouseCallback('Camera', souris)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    elif key==ord('+'):
        color+=1
        lo[0]=color-5
        hi[0]=color+5
    elif key==ord('-'):
        color-=1
        lo[0]=color-5
        hi[0]=color+5
cap.release()
cv2.destroyAllWindows()
