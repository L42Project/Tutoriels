import cv2
import numpy as np
from matplotlib import pyplot as plt

objet=0
nbr_classes=180
seuil=30
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0)

def click(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, roi_hist, frame, objet

    if event==cv2.EVENT_LBUTTONDBLCLK:
        roi_x, roi_y, roi_w, roi_h=cv2.selectROI('ROI', frame, False, False)
        roi=frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        roi_x-=30
        roi_y-=30
        roi_w+=60
        roi_h+=60
        hsv_roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist=cv2.calcHist([hsv_roi], [0], None, [nbr_classes], [0, nbr_classes])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        cv2.destroyWindow('ROI')
        plt.clf()
        plt.plot(roi_hist)
        plt.show(block=False)
        plt.pause(0.01)
        objet=1

video=cv2.VideoCapture("foot.webm")

cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', click)
while True:
    ret, frame=video.read()
    frame=cv2.resize(frame, (frame.shape[1]*3, frame.shape[0]*3))[200:1000, 400:2400] # A adapter selon la vidéo ou enlever !!!!

    if objet:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask=cv2.calcBackProject([hsv], [0], roi_hist, [0, nbr_classes], 1)

        _, mask=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)
        mask=cv2.erode(mask, None, iterations=3)
        mask=cv2.dilate(mask, None, iterations=3)

        _, rect=cv2.meanShift(mask, (roi_x, roi_y, roi_w, roi_h), term_criteria)
        roi_x, roi_y, w, h=rect

        if np.sum(mask[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w])==0:
            objet=0
        cv2.putText(frame, "Marty McFly", (roi_x-40, roi_y-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.rectangle(mask, (roi_x, roi_y), (roi_x + w, roi_y + h), (255, 255, 255), 2)

        cv2.imshow("Mask", mask)
        cv2.putText(frame, "seuil[p|m]: {:d}".format(seuil), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

    cv2.imshow("Camera", frame)
    key=cv2.waitKey(50)&0xFF
    if key==ord('q'):
        quit()
    if key==ord('p'):
        seuil=min(250, seuil+1)
    if key==ord('m'):
        seuil=max(1, seuil-1)
    if key==ord('a'):
        for cpt in range(1800):
            ret, frame=video.read()

video.release()
cv2.destroyAllWindows()
