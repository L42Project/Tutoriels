import numpy as np
import matplotlib.pyplot as plt

def fonction(x):
    return x**2+3*x-2

def gradient_fonction(x):
    return 2*x+3

xvals=np.arange(-5, 3, 0.1)
yvals=fonction(xvals)
plt.plot(xvals, yvals)

x=np.random.random_integers(-4, 3)+np.random.rand(1)[0]
lr=0.2
i=0
while True:
    plt.scatter(x, fonction(x), color='#FF0000')
    plt.draw()
    plt.pause(0.5)
    x=x-lr*gradient_fonction(x)
    print("itération {:3d}  -> x={:+7.5f}".format(i, x))
    i+=1
