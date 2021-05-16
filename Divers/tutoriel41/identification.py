import face_recognition
import cv2
import numpy as np
import os
import time

class identify:
    font=cv2.FONT_HERSHEY_DUPLEX
    color=(255, 255, 255)

    def __init__(self, entree, embedding_file, name_file, width_max=320, tolerance=0.5):
        self.entree=entree
        self.width_max=width_max
        self.tolerance=tolerance

        if not os.path.exists(embedding_file):
            print("Fichier", embedding_file, "non trouvé")
            quit()
        if not os.path.exists(name_file):
            print("Fichier", name_file, "non trouvé")
            quit()
        self.known_face_encodings=np.load(embedding_file)
        self.known_face_names=np.load(name_file)

        if entree.split(':')[0]=="https":
            import pafy
            video=pafy.new(entree)
            video_mp4=video.getbest(preftype="mp4")
            self.video_capture=cv2.VideoCapture(video_mp4.url)
        elif entree=="csi":
            self.video_capture=cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");        
        elif entree=="realsense":
            import pyrealsense2 as rs
            self.pipeline=rs.pipeline()
            config=rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.pipeline.start(config)
            self.video_capture=None
        elif entree.split(':')[0]=="file":
            fichier=entree.split(':')[1]
            if not os.path.exists(fichier):
                print("Fichier", fichier, "non trouvé")
                quit()
            self.video_capture=cv2.VideoCapture(fichier)
        else:
            print("Entree inconnue")
            quit()

    def read(self):
        if self.video_capture is not None:
            ret, self.frame=self.video_capture.read()
        else:
            while True:
                frames=self.pipeline.wait_for_frames()
                color_frame=frames.get_color_frame()
                if not color_frame:
                    continue
                break
            self.frame=np.array(color_frame.get_data())

    def analyse(self):
        frame=self.frame
        if frame.shape[1]>self.width_max:
            self.ratio=self.width_max/frame.shape[1]
            frame_to_analyse=cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)
        else:
            self.ratio=1
            frame_to_analyse=frame
            
        self.face_locations=face_recognition.face_locations(frame_to_analyse)
        face_encodings=face_recognition.face_encodings(frame_to_analyse, self.face_locations)

        self.face_names=[]
        self.face_distances=[]
        for face_encoding in face_encodings:
            distances=face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if np.min(distances)<self.tolerance:
                best_match_index=np.argmin(distances)
                name=self.known_face_names[best_match_index]
            else:
                name="Inconnu"
            self.face_names.append(name)
            self.face_distances.append(np.min(distances))
        
    def render(self):
        self.frame_render=self.frame
        for (top, right, bottom, left), name, distance in zip(self.face_locations, self.face_names, self.face_distances):
            top   =int(top    /self.ratio)
            right =int(right  /self.ratio)
            bottom=int(bottom /self.ratio)
            left  =int(left   /self.ratio)
            cv2.rectangle(self.frame_render, (left, top), (right, bottom), self.color, 2)
            msg="[{:4.2f}] {}".format(distance, name)
            cv2.putText(self.frame_render, msg, (left, int(bottom+(bottom-top)*0.2)), self.font, 0.8, self.color, 1)
        
#toto=identify("realsense", "face_encodings.npy", "face_names.npy")
toto=identify("https://www.youtube.com/watch?v=m_xWEofOqHI", "face_encodings.npy", "face_names.npy")
#toto=identify("csi", "face_encodings.npy", "face_names.npy")
#toto=identify("file:photo.jpg", "face_encodings.npy", "face_names.npy")

while True:
    toto.read()
    toto.analyse()
    print("#############")
    for name, distance in zip(toto.face_names, toto.face_distances):
        print("   ", name, distance)
    toto.render()
    cv2.imshow("Frame render", toto.frame_render)

    key=cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
    if key==ord('a'):
        for i in range(100):
            toto.read()
    if key==ord('z'):
        for i in range(2000):
            toto.read()

cv2.destroyAllWindows()
