import cv2
import numpy as np
from matplotlib import pyplot as plt

objet=0
nbr_classes=180
seuil=30
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0)
mode=0

def click(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, roi_hist, frame, objet

    if event==cv2.EVENT_LBUTTONDBLCLK:
        roi_x, roi_y, roi_w, roi_h=cv2.selectROI('ROI', frame, False, False)
        roi=frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        hsv_roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist=cv2.calcHist([hsv_roi], [0], None, [nbr_classes], [0, nbr_classes])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        cv2.destroyWindow('ROI')
        plt.clf()
        plt.plot(roi_hist)
        plt.show(block=False)
        plt.pause(0.01)
        objet=1

video=cv2.VideoCapture(0)
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', click)
while True:
    ret, frame=video.read()
    if objet:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask=cv2.calcBackProject([hsv], [0], roi_hist, [0, nbr_classes], 1)
        _, mask=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)
        mask=cv2.erode(mask, None, iterations=3)
        mask=cv2.dilate(mask, None, iterations=3)
        if mode:
            _, rect=cv2.CamShift(mask, (roi_x, roi_y, roi_w, roi_h), term_criteria)
            pts=cv2.boxPoints(_)
            pts=np.int0(pts)
            img2=cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        else:
            _, rect=cv2.meanShift(mask, (roi_x, roi_y, roi_w, roi_h), term_criteria)
        roi_x, roi_y, w, h=rect
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + w, roi_y + h), (255, 255, 255), 2)
        cv2.imshow("Mask", mask)
        cv2.putText(frame, "seuil[p|m]: {:d}   mode[o]:{}".format(seuil, "CamShift" if mode else "meanshift"), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

    cv2.imshow("Camera", frame)
    key=cv2.waitKey(10)&0xFF
    if key==ord('q'):
        quit()
    if key==ord('p'):
        seuil=min(250, seuil+1)
    if key==ord('m'):
        seuil=max(1, seuil-1)
    if key==ord('o'):
        mode=not mode

video.release()
cv2.destroyAllWindows()
