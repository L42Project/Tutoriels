import cv2
import numpy as np
import operator

marge=4
case=28+2*marge
taille_grille=9*case

methode=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
v1=9

cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (5, 5), 0)
    thresh=cv2.adaptiveThreshold(gray, 255, methode, cv2.THRESH_BINARY_INV, v1, 2)
    #cv2.imshow("thresh", thresh)
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grille=None
    maxArea=0
    for c in contours:
        area=cv2.contourArea(c)
        if area>25000:
            peri=cv2.arcLength(c, True)
            polygone=cv2.approxPolyDP(c, 0.01*peri, True)
            if area>maxArea and len(polygone)==4:
                contour_grille=polygone
                maxArea=area
    if contour_grille is not None:
        cv2.drawContours(frame, [contour_grille], 0, (0, 255, 0), 2)
        points=np.vstack(contour_grille).squeeze()
        points=sorted(points, key=operator.itemgetter(1))
        if points[0][0]<points[1][0]:
            if points[3][0]<points[2][0]:
                pts1=np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1=np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0]<points[2][0]:
                pts1=np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1=np.float32([points[1], points[0], points[2], points[3]])
        pts2=np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [taille_grille, taille_grille]])
        M=cv2.getPerspectiveTransform(pts1, pts2)
        grille=cv2.warpPerspective(frame, M, (taille_grille, taille_grille))
        cv2.putText(frame, "1", (points[0][0], points[0][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)
        cv2.putText(frame, "2", (points[1][0], points[1][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)
        cv2.putText(frame, "3", (points[2][0], points[2][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)
        cv2.putText(frame, "4", (points[3][0], points[3][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)
        cv2.imshow("grille", grille)
    txt="ADAPTIVE_THRESH_MEAN_C" if methode==cv2.ADAPTIVE_THRESH_MEAN_C else "ADAPTIVE_THRESH_GAUSSIAN_C"
    cv2.putText(frame, "[p|m]v1: {:2d}  [o]methode: {}".format(v1, txt), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)
    cv2.imshow("frame", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('p'):
        v1=min(21, v1+2)
    if key==ord('m'):
        v1=max(3, v1-2)
        print(v1)
    if key==ord('o'):
        if methode==cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
            methode=cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            methode=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
cap.release()
cv2.destroyAllWindows()
