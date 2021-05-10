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

pipeline_wrapper=rs.pipeline_wrapper(pipeline)
pipeline_profile=config.resolve(pipeline_wrapper)
device=pipeline_profile.get_device()

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
    print("Temps", time.time()-start)
    
    display.Render(cuda_image)
    cuda_image=jetson.utils.cudaToNumpy(cuda_image)

    cv2.imshow('RealSense1', depth_colormap)
    #cv2.imshow('RealSense2', color_image)
    cv2.imshow('cuda_image', cuda_image)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        pipeline.stop()
        quit()
