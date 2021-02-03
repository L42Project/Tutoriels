from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

numPoints=8
radius=3

textures=['gazon', 'gravier', 'bois']

for texture in textures:
	print(">>>", "{}.jpg".format(texture))
	image=cv2.imread("{}.jpg".format(texture), 0)
	if image is None:
		quit("Probleme image...")
	lbp=feature.local_binary_pattern(image, numPoints, radius, method='default')
	hist_ref, _=np.histogram(lbp, bins=2**numPoints, range=(0, 2**numPoints))

	cv2.imshow("Image", image)
	cv2.imshow("LBP", lbp)
	plt.plot(hist_ref)
	plt.show()

	key=cv2.waitKey(1)&0xFF
	if key==ord('q'):
		quit()
