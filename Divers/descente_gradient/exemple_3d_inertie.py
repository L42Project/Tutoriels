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
#ax.contour(X, Y, Z, 70, rstride=1, cstride=1, cmap='plasma')

plt.xlabel("Paramètre 1 (x)")
plt.ylabel("Paramètre 2 (y)")

x=np.random.random_integers(-2, 2)+np.random.rand(1)[0]
y=np.random.random_integers(-2, 2)+np.random.rand(1)[0]

lr=0.2
lr2=0.9
correction_x=0
correction_y=0
i=0
while True:
    g_x, g_y=gradient_fonction(x, y)
    correction_x=lr2*correction_x-lr*g_x
    x=x+correction_x
    correction_y=lr2*correction_y-lr*g_y
    y=y+correction_y
    ax.scatter(x, y, fonction(x, y), marker='o', s=10, color='#FF0000')
    plt.draw()
    print("itération {:3d}  -> x={:+7.5f} y={:+7.5f}".format(i, x, y))
    plt.pause(0.05)
    i+=1
