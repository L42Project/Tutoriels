import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import cv2
import glob

k=5
cluster_std=1.30
n_samples=300
X, y=make_blobs(n_samples=n_samples, centers=k, cluster_std=cluster_std)

fig, (ax1, ax2)=plt.subplots(1, 2)
canvas=FigureCanvas(fig)
fig.set_size_inches(10, 6)

while 1:
    ax1.cla()
    ax1.scatter(X[:,0], X[:,1], marker='+', c="#FF0000")
    ax1.set_title('Données')

    wcss=[]
    for i in range(1, 11):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    ax2.cla()
    ax2.plot(range(1, 11), wcss, c="#FF0000")
    ax2.set_title('WCSS pour "elbow method"')

    canvas.draw()
    img=np.array(canvas.renderer.buffer_rgba())
    cv2.putText(img, "[r] reset  [q] quit".format(k), (450, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow("plot", img)
    key=cv2.waitKey()&0xFF
    if key==ord('r'):
        X, y=make_blobs(n_samples=n_samples, centers=k, cluster_std=cluster_std)
    if key==ord('q'):
        quit()
