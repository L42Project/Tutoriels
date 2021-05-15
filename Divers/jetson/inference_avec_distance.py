import pyrealsense2 as rs
import cv2
import numpy as np
import jetson.inference
import jetson.utils
import time

net=jetson.inference.detectNet("SSD-Inception-v2", threshold=0.5)
#net=jetson.inference.detectNet("SSD-MobileNet-v2", threshold=0.5)
display=jetson.utils.videoOutput("display://0")

pipeline=rs.pipeline()
config=rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)

while True:

    frames=pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    depth_frame=aligned_frames.get_depth_frame()
    color_frame=aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth_image=np.array(depth_frame.get_data())
    color_image=np.array(color_frame.get_data())

    depth_colormap=cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    start=time.time()
    cuda_image=jetson.utils.cudaFromNumpy(color_image)    
    detections=net.Detect(cuda_image, color_image.shape[1], color_image.shape[0])
    
    print("#######################################")
    print("Temps", time.time()-start)
    for detection in detections:
        print(detection)
        x, y=detection.Center
        x1=int(x-detection.Width/2)
        y1=int(y-detection.Height/2)
        x2=int(x+detection.Width/2)
        y2=int(y+detection.Height/2)
        if detection.ClassID==1:
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            dist=depth_frame.get_distance(int(x), int(y))
            if dist<1:
                msg="{:2.0f} cm".format(dist*100)
            else:
                msg="{:4.2f} m".format(dist)
            cv2.putText(color_image, msg, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
    cv2.imshow('RealSense1', depth_colormap)
    cv2.imshow('RealSense2', color_image)

    display.Render(cuda_image)
    cuda_image=jetson.utils.cudaToNumpy(cuda_image)
    #display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    cv2.imshow('cuda_image', cuda_image)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        pipeline.stop()
        quit()
