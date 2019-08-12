import cv2
import numpy as np
from matplotlib import pyplot as plt

objet=0
nbr_classes=180
seuil=30
v1=0

def click(event, x, y, flags, param):
    global roi_hist, frame, objet
    if event==cv2.EVENT_LBUTTONDBLCLK:
        roi_x, roi_y, roi_w, roi_h=cv2.selectROI('ROI', frame, False, False)
        roi=frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        hsv_roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist=cv2.calcHist([hsv_roi], [0, 1], None, [nbr_classes, 256], [0, nbr_classes, 0, 256])
        cv2.normalize(roi_hist[0, :], roi_hist[0, :], 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(roi_hist[1, :], roi_hist[1, :], 0, 255, cv2.NORM_MINMAX)
        cv2.destroyWindow('ROI')
        plt.clf()
        plt.plot(roi_hist[0, :])
        plt.plot(roi_hist[1, :])
        plt.show(block=False)
        plt.pause(0.01)
        objet=1

video=cv2.VideoCapture(0)
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', click)
while True:
    ret, frame=video.read()
    if ret is False:
        quit()

    if objet:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask=cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, nbr_classes, v1, 256], 1)

        _, mask2=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)
        mask2=cv2.erode(mask2, None, iterations=3)
        mask2=cv2.dilate(mask2, None, iterations=5)

        elements=cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(elements) > 0:
           c=max(elements, key=cv2.contourArea)
           x,y,w,h=cv2.boundingRect(c)
           cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

        cv2.imshow("Mask", mask)
        cv2.imshow("Mask2", mask2)
        image2=cv2.bitwise_and(frame, frame, mask=mask2)
        cv2.imshow("Image2", image2)
        cv2.putText(frame, "seuil[p|m]: {:d}  v1[o|l]: {:d}".format(seuil, v1), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

    cv2.imshow("Camera", frame)
    key=cv2.waitKey(5)
    if key==ord('q'):
        quit()
    if key==ord('p'):
        seuil=min(250, seuil+1)
    if key==ord('m'):
        seuil=max(1, seuil-1)
    if key==ord('o'):
        v1=min(250, v1+1)
    if key==ord('l'):
        v1=max(0, v1-1)

video.release()
cv2.destroyAllWindows()
