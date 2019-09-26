import numpy as np
import matplotlib.pyplot as plt

def fonction(x):
    return 3*x**4-4*x**3-12*x**2-0*x-3

def gradient_fonction(x):
    return 12*x**3-12*x**2-24*x

xvals=np.arange(-3, 4, 0.1)
yvals=fonction(xvals)
plt.plot(xvals, yvals)

x=np.random.random_integers(-3, 3)+np.random.rand(1)[0]
i=0
print("itération: {}  x={}".format(i, x))
lr=0.015
while True:
    plt.scatter(x, fonction(x), color='#FF0000')
    plt.draw()
    plt.pause(0.5)
    x=x-lr*gradient_fonction(x)
    i+=1
    print("itération {:3d}  -> x={}".format(i, x))
