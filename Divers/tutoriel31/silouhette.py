import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import cv2

cluster_std=1.30
n_samples=300
X, y=make_blobs(n_samples=n_samples, centers=5, cluster_std=cluster_std)

fig, (ax1, ax2)=plt.subplots(1, 2)
canvas=FigureCanvas(fig)
fig.set_size_inches(11, 6)

k=2
while 1:
    ax1.cla()
    ax1.scatter(X[:,0], X[:,1], marker='+', c="#FF0000")

    kmeans=KMeans(n_clusters=k)
    pred_y=kmeans.fit_predict(X)

    ax2.cla()
    ax2.scatter(X[:,0], X[:,1], c=pred_y, marker='+')
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='#0000FF')
    canvas.draw()

    img=np.array(canvas.renderer.buffer_rgba())
    cv2.putText(img, "Nbr cluster={:02d}    [p|m] nbr clusters   [r] reset  [q] quit".format(k), (250, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow("plot", img)
    key=cv2.waitKey()&0xFF
    if key==ord('p'):
        k=min(99, k+1)
    if key==ord('m'):
        k=max(2, k-1)
    if key==ord('r'):
        X, y=make_blobs(n_samples=n_samples, centers=5, cluster_std=cluster_std)
    if key==ord('q'):
        quit()
