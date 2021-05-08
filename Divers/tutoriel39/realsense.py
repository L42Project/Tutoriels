import pyrealsense2 as rs
import numpy as np
import cv2

lo=np.array([95, 100, 50])
hi=np.array([105, 255, 255])
color_infos=(0, 255, 255)

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
    
    cv2.imshow('RealSense1', depth_colormap)

    frame=color_image
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(image, lo, hi)
    image=cv2.blur(image, (7, 7))
    mask=cv2.erode(mask, None, iterations=4)
    mask=cv2.dilate(mask, None, iterations=4)
    image2=cv2.bitwise_and(frame, frame, mask=mask)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c=max(elements, key=cv2.contourArea)
        ((x, y), rayon)=cv2.minEnclosingCircle(c)
        if rayon>30:
            cv2.circle(image2, (int(x), int(y)), int(rayon), color_infos, 2)
            cv2.circle(frame, (int(x), int(y)), 5, color_infos, 10)
            cv2.line(frame, (int(x), int(y)), (int(x)+150, int(y)), color_infos, 2)
            print(">>>", depth_colormap[int(y), int(x)])
            dist=depth_frame.get_distance(int(x), int(y))
            if dist<1:
                msg="{:2.0f} cm".format(dist*100)
            else:
                msg="{:4.2f} m".format(dist)
            cv2.putText(frame, msg, (int(x)+10, int(y) -10), cv2.FONT_HERSHEY_DUPLEX, 1, color_infos, 1, cv2.LINE_AA)
    cv2.imshow('Camera', frame)
    
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        pipeline.stop()
        quit()
