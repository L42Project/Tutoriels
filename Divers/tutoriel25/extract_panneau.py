import cv2
import os
import numpy as np
import random
import common

video_dir="D:\dashcam Cedric"

l=os.listdir(video_dir)

for video in l:
    if not video.endswith("mp4"):
        continue
    cap=cv2.VideoCapture(video_dir+"/"+video)

    print("video:", video)
    while True:
        ret, frame=cap.read()
        if ret is False:
            break
        f_w, f_h, f_c=frame.shape
        frame=cv2.resize(frame, (int(f_h/1.5), int(f_w/1.5)))

        image=frame[200:400, 700:1000]
        cv2.rectangle(frame, (700, 200), (1000, 400), (255, 255, 255), 1)

        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=60, minRadius=5, maxRadius=45)
        if circles is not None:
            circles=np.int16(np.around(circles))
            for i in circles[0,:]:
                if i[2]!=0:
                    panneau=cv2.resize(image[max(0, i[1]-i[2]):i[1]+i[2], max(0, i[0]-i[2]):i[0]+i[2]], (common.size, common.size))/255
                    cv2.imshow("panneau", panneau)
        cv2.putText(frame, "fichier:"+video, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Video", frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            quit()
        if key==ord('a'):
            for cpt in range(100):
                ret, frame=cap.read()
        if key==ord('f'):
            break

cv2.destroyAllWindows()
