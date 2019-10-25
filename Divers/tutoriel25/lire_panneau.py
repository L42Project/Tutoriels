import tensorflow as tf
import cv2
import os
import numpy as np
import random
import common

th1=30
th2=55

video_dir="dashcam Cedric"

tab_panneau, tab_image_panneau=common.lire_images_panneaux(common.dir_images_panneaux)

model_is_panneau=common.is_panneau_model()
checkpoint=tf.train.Checkpoint(model_is_panneau=model_is_panneau)
checkpoint.restore(tf.train.latest_checkpoint("./training_is_panneau/"))

model_panneau=common.panneau_model(len(tab_panneau))
checkpoint=tf.train.Checkpoint(model_panneau=model_panneau)
checkpoint.restore(tf.train.latest_checkpoint("./training_panneau/"))

l=os.listdir(video_dir)
random.shuffle(l)

for video in l:
#for video in ["20190918_173518_EF.mp4"]:
    if not video.endswith("mp4"):
        continue
    cap=cv2.VideoCapture(video_dir+"/"+video)

    print("video:", video)
    id_panneau=-1    
    while True:
        ret, frame=cap.read()
        if ret is False:
            break
        f_w, f_h, f_c=frame.shape
        frame=cv2.resize(frame, (int(f_h/1.5), int(f_w/1.5)))

        image=frame[200:400, 700:1000]
        cv2.rectangle(frame, (700, 200), (1000, 400), (255, 255, 255), 1)

        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img=cv2.medianBlur(image,5)

        circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=th1, param2=th2, minRadius=5, maxRadius=45)
        if circles is not None:
            circles=np.int16(np.around(circles))
            for i in circles[0,:]:
                if i[2]!=0:
                    panneau=cv2.resize(image[max(0, i[1]-i[2]):i[1]+i[2], max(0, i[0]-i[2]):i[0]+i[2]], (common.size, common.size))/255
                    cv2.imshow("panneau", panneau)
                    prediction=model_is_panneau(np.array([panneau]), training=False)
                    print("prediction", prediction)
                    if prediction[0][0]>0.9:
                        prediction=model_panneau(np.array([panneau]), training=False)
                        id_panneau=np.argmax(prediction[0])
                        print("panneau", prediction, id_panneau, tab_panneau[id_panneau])
                        w, h, c=tab_image_panneau[id_panneau].shape
        if id_panneau!=-1:
            frame[0:h, 0:w, :]=tab_image_panneau[id_panneau]
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
