from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

cap=cv2.VideoCapture(0)
numPoints=24
radius=3
image_ref=None

width=320
height=240

windowsize_r=30
windowsize_c=30

rows=int(height/windowsize_r)
cols=int(width/windowsize_c)

tab_score=np.empty((rows, cols), dtype=np.float32)
id_method_distance=6
id_method_lbp=2
seuil=0.3

while True:
	ret, frame=cap.read()
	image=cv2.resize(frame, (width, height))

	if image_ref is not None:
		gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for r in range(rows):
			for c in range(cols):
				window=gray[r*windowsize_r:(r+1)*windowsize_r, c*windowsize_c:(c+1)*windowsize_c]
				lbp=feature.local_binary_pattern(window, numPoints, radius, method=method_lbp[id_method_lbp])
				hist, _=np.histogram(lbp, normed=True, bins=numPoints, range=(0, numPoints))
				score=cv2.compareHist(hist.astype(np.float32), hist_ref.astype(np.float32), method_distance[id_method_distance])
				tab_score[r, c]=score

	if image_ref is not None:
		tab_score2=np.zeros_like(tab_score)
		tab_score2[tab_score<seuil]=1
		cv2.imshow("SCORE2", cv2.resize(tab_score2, (int(width), int(height)), interpolation = cv2.INTER_NEAREST))
		tab_score=np.clip(1/(10*tab_score), 0., 1.)
		cv2.imshow("Score", cv2.resize(tab_score, (int(width), int(height)), interpolation = cv2.INTER_NEAREST))

	cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (100, 100, 100), cv2.FILLED)
	txt="[q]Quit   [p|m]Seuil={:2.1f} [d]Methode distance:{}   [l]Methode LBP:{}".format(seuil, method_distance[id_method_distance], method_lbp[id_method_lbp])
	cv2.putText(frame, txt, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	cv2.imshow("Image", frame)

	key=cv2.waitKey(1)&0xFF
	if key==ord('q'):
		quit()
	if key==ord('i'):
		image_ref=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		lbp_ref=feature.local_binary_pattern(image_ref, numPoints, radius, method=method_lbp[id_method_lbp])
		hist_ref, _=np.histogram(lbp_ref, normed=True, bins=numPoints, range=(0, numPoints))
	if key==ord('l'):
		id_method_lbp=(id_method_lbp+1)%len(method_lbp)
		if image_ref is not None:
			lbp_ref=feature.local_binary_pattern(image_ref, numPoints, radius, method=method_lbp[id_method_lbp])
			hist_ref, _=np.histogram(lbp_ref, normed=True, bins=numPoints, range=(0, numPoints))
	if key==ord('d'):
		id_method_distance=(id_method_distance+1)%len(method_distance)
	if key==ord('p'):
		seuil+=0.1
	if key==ord('m'):
		seuil=max(0.1, seuil-0.1)
