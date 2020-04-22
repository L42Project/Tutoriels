import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
import cv2
import glob

k=5
cluster_std=1.30
n_samples=300
fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2, 2)
canvas=FigureCanvas(fig)
fig.set_size_inches(12, 8)

X, y=make_blobs(n_samples=n_samples, centers=k, cluster_std=cluster_std)

while 1:
    ax1.cla()
    ax1.plot(X[:,0], X[:,1], "+", c="#FF0000")
    ax1.set_title('Données')

    wcss=[]
    tab_silhouette=[]
    for i in range(2, 11):
        kmeans=KMeans(n_clusters=i)
        cluster_labels=kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        tab_silhouette.append(silhouette_score(X, cluster_labels))

    ax2.cla()
    ax2.plot(range(2, 11), wcss, c="#FF0000")
    ax2.set_title('WCSS pour "elbow method"')

    ax3.cla()
    ax3.plot(range(2, 11), tab_silhouette, c="#FF0000")
    ax3.set_title('Coefficient silhouette')

    kmeans=KMeans(n_clusters=np.argmax(tab_silhouette)+2)
    pred_y=kmeans.fit_predict(X)
    ax4.cla()
    ax4.scatter(X[:,0], X[:,1], c=pred_y, marker='+')
    ax4.set_title('Données + centre clusters')
    ax4.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c="#0000FF")

    canvas.draw()
    img=np.array(canvas.renderer.buffer_rgba())
    cv2.putText(img, "[r] reset  [q] quit".format(k), (450, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow("plot", img)
    key=cv2.waitKey()&0xFF
    if key==ord('r'):
        X, y=make_blobs(n_samples=n_samples, centers=k, cluster_std=cluster_std)
    if key==ord('q'):
        quit()
