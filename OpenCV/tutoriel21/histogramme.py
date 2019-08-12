import cv2
import numpy as np
from matplotlib import pyplot as plt

video=cv2.VideoCapture(0)
mode=0
nbr_classes=180

while True:
    ret, frame=video.read()
    if ret is False:
        quit()

    if mode:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist=cv2.calcHist([frame], [0], None, [nbr_classes], [0, 256])
        label="Intensité"
    else:
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv], [0], None, [nbr_classes], [0, 180])
        label="Couleur"
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    cv2.putText(frame, "Mode[o] {:d}  Nbr classes[p|m]: {:d}".format(mode, nbr_classes), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    plt.clf()
    plt.plot(hist)
    plt.title(label)
    plt.show(block=False)
    cv2.imshow("image", frame)
    plt.pause(0.001)

    key=cv2.waitKey(5)&0xFF
    if key==ord('q'):
        quit()
    if key==ord('o'):
        mode=not mode
        nbr_classes=(255 if mode else 180)
    if key==ord('p'):
        nbr_classes*=2
    if key==ord('m'):
        nbr_classes//=2

video.release()
cv2.destroyAllWindows()
