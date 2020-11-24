import cv2
import operator
import picamera
import picamera.array

face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
profile_cascade=cv2.CascadeClassifier("./haarcascade_profileface.xml")
marge=70
WIDTH=640
HEIGHT=480

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution=(WIDTH, HEIGHT)            
        while True:
            camera.capture(stream, 'bgr', use_video_port=True)
            tab_face=[]
            tickmark=cv2.getTickCount()
            gray=cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)
            face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(5, 5))
            for x, y, w, h in face:
                tab_face.append([x, y, x+w, y+h])
                face=profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
                for x, y, w, h in face:
                    tab_face.append([x, y, x+w, y+h])
                gray2=cv2.flip(gray, 1)
                face=profile_cascade.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=4)
                for x, y, w, h in face:
                    tab_face.append([WIDTH-x, y, WIDTH-(x+w), y+h])
                tab_face=sorted(tab_face, key=operator.itemgetter(0, 1))
                index=0
                for x, y, x2, y2 in tab_face:
                    if not index or (x-tab_face[index-1][0]>marge or y-tab_face[index-1][1]>marge):
                        cv2.rectangle(stream.array, (x, y), (x2, y2), (0, 0, 255), 2)
                    index+=1
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
            cv2.putText(stream.array, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow('video', stream.array)
            stream.seek(0)
            stream.truncate()            
cv2.destroyAllWindows()
