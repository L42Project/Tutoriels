import cv2
import numpy as np

stop=0
cap=cv2.VideoCapture("autoroute.mp4")
while True:
    if not stop:
        ret, frame=cap.read()
        if ret is False:
            quit()
        image=frame.copy()

    cv2.imshow("image", image)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grad_x=cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=5)
    cv2.imshow("grad x", grad_x)

    grad_y=cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=5)
    cv2.imshow("grad y", grad_y)

    if not stop:
        key=cv2.waitKey(10)&0xFF
    else:
        key=cv2.waitKey()
        image=frame.copy()

    if key==ord('q'):
        break
    if key==ord('s'):
        stop=not stop
    if key==ord('a'):
        for cpt in range(200):
            ret, frame=cap.read()
            image=frame.copy()
cap.release()
cv2.destroyAllWindows()
