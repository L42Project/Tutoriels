import cv2

object_cascade=cv2.CascadeClassifier("./cars.xml")
cap=cv2.VideoCapture('cars.mp4')

clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=clahe.apply(gray)
    object=object_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=2)
    for x, y, w, h in object:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
        cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('gray', gray)
        cv2.imshow('video', frame)
        if cv2.waitKey(1)==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
                                                                    
