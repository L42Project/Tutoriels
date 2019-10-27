from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import math

def fonction(X, Y):
    return X*np.exp(-X**2-Y**2)+(X**2+Y**2)/20

def gradient_fonction(X, Y):
    g_x=np.exp(-X**2-Y**2)+X*-2*X*np.exp(-X**2-Y**2)+X/10
    g_y=-2*Y*X*np.exp(-X**2-Y**2)+Y/10
    return g_x, g_y

fig=plt.figure()
fig.set_size_inches(9, 7, forward=True)
ax=Axes3D(fig, azim=-29, elev=49)
X=np.arange(-3, 3, 0.2)
Y=np.arange(-3, 3, 0.2)
X, Y=np.meshgrid(X, Y)
Z=fonction(X, Y)
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
plt.xlabel("Paramètre 1 (x)")
plt.ylabel("Paramètre 2 (y)")

x1=x2=np.random.random_integers(-2, 2)+np.random.rand(1)[0]
y1=y2=np.random.random_integers(-2, 2)+np.random.rand(1)[0]

lr=0.2
lr2=0.9
correction_x1=0
correction_y1=0
i=0
while True:
    g_x1, g_y1=gradient_fonction(x1, y1)
    g_x2, g_y2=gradient_fonction(x2, y2)

    correction_x1=lr2*correction_x1-lr*g_x1
    x1=x1+correction_x1
    correction_y1=lr2*correction_y1-lr*g_y1
    y1=y1+correction_y1

    x2=x2-lr*g_x2
    y2=y2-lr*g_y2

    ax.scatter(x1, y1, fonction(x1, y1), marker='o', s=10, color='#FF0000')
    ax.scatter(x2, y2, fonction(x2, y2), marker='o', s=10, color='#00FF00')
    plt.draw()
    print("iteration= {} x1={:+7.5f} y1={:+7.5f} x2={:+7.5f} y2={:+7.5f}".format(i, x1, y1, x2, y2))
    plt.pause(0.05)
    i+=1
