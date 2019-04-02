import numpy as np
import cv2
import picamera
import picamera.array

WIDTH=640
HEIGHT=480

face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution=(WIDTH, HEIGHT)
        while True:
            camera.capture(stream, 'bgr', use_video_port=True)
            tickmark=cv2.getTickCount()
            gray=cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)
            face=face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
            for x, y, w, h in face:
                cv2.rectangle(stream.array, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
            cv2.putText(stream.array, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow('video', stream.array)
            stream.seek(0)
            stream.truncate()
cv2.destroyAllWindows()
