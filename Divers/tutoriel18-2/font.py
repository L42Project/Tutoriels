from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np

for i in range(1, 10):                
    image=Image.new("L", (28, 28))
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype("din1451altG.ttf", 27)
    text="{:d}".format(i)
    draw.text((10, 0), text, font=font, fill=(255))
    image=np.array(image).reshape(28, 28, 1)
    cv2.imshow("image", image)
    key=cv2.waitKey()
    if key&0xFF==ord('q'):
        quit()
                                                                    

    
