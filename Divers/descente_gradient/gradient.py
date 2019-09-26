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
ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
plt.xlabel("Paramètre 1 (x)")
plt.ylabel("Paramètre 2 (y)")

x, y=np.meshgrid(np.arange(-3, 3, 0.2),
                np.arange(-3, 3, 0.2))
z=-1

u, v=gradient_fonction(x, y)
w=0
ax.quiver(x, y, z, u, v, w, length=0.15, normalize=True, color='#333333')

plt.show()
