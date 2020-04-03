import tensorflow as tf
import cv2
import numpy as np
import config

my_model=tf.keras.models.load_model('saved_model\\my_model')

cap=cv2.VideoCapture(0)
width=cap.get(3)
height=cap.get(4)

while True:
    ret, frame=cap.read()
    img=cv2.resize(frame, (config.size, config.size))/255
    img=np.array([img], dtype=np.float32)
    prediction=my_model.predict(img)
    if prediction[0][0]>0.3:
        color=(0, 255, 0)
    else:
        color=(0, 0, 255)
    cv2.rectangle(frame, (0, int(height)-30), (int(width*prediction[0][0]), int(height)), color, cv2.FILLED)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        quit()
