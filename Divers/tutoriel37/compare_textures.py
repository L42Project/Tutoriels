from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

method_distance=[cv2.HISTCMP_CORREL,
				 cv2.HISTCMP_CHISQR,
			  	 cv2.HISTCMP_INTERSECT,
				 cv2.HISTCMP_BHATTACHARYYA,
				 cv2.HISTCMP_HELLINGER,
				 cv2.HISTCMP_CHISQR_ALT,
				 cv2.HISTCMP_KL_DIV]

method_lbp=['default',
			'ror',
			'uniform',
			'var']

numPoints=8
radius=3
id_method_distance=6
id_method_lbp=2

textures=['gazon', 'gravier', 'bois']

tab_images=[]
tab_hist=[]
for texture in textures:
	print(">>>", "{}.jpg".format(texture))
	image=cv2.imread("{}.jpg".format(texture), 0)
	if image is None:
		quit("Probleme image...")
	lbp=feature.local_binary_pattern(image, numPoints, radius, method=method_lbp[id_method_lbp])
	hist_ref, _=np.histogram(lbp, normed=True, density=True, bins=2**numPoints, range=(0, 2**numPoints))
	tab_images.append(image)
	tab_hist.append(hist_ref)

for fichier in glob.glob("textures/*.jpg"):
	print("Lecture de", fichier)
	frame=cv2.imread(fichier)

	if frame is None:
		print("Probleme image ...")
		continue

	cv2.imshow("Image", frame)

	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	tab_distance=[]
	for id in range(len(tab_hist)):
		h=tab_hist[id]
		lbp=feature.local_binary_pattern(gray, numPoints, radius, method=method_lbp[id_method_lbp])
		hist, _=np.histogram(lbp, normed=True, density=True, bins=2**numPoints, range=(0, 2**numPoints))
		score=cv2.compareHist(hist.astype(np.float32), h.astype(np.float32), method_distance[id_method_distance])
		tab_distance.append(score)
		print("  score {:10}: {:2.6f}".format(textures[id], score))

	tab_distance=np.array(tab_distance)
	print("  -> texture:", textures[np.argmin(tab_distance)])

	key=cv2.waitKey()&0xFF
	if key==ord('q'):
		quit()
