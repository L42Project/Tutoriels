import cv2
import numpy as np
import common

image=common.moyenne_image('autoroute.mp4', 100)
cv2.imshow('fond', image.astype(np.uint8))
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()
