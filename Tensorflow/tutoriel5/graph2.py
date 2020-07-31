import numpy as np
import matplotlib.pyplot as plot
from L42Project import ia as LPia
import sys

fig=plot.figure()
plot.ylim(0, 0.35)
ax=fig.gca()
ax.set_yticks(np.arange(0, 0.4, 0.05), True)
vals = ax.get_yticks()
ax.set_yticklabels(['{:.0%}'.format(x) for x in vals])
#ax2=ax.twinx()
#ax2.set_yticks(np.arange(0, 1.1, 0.1), True)
#vals=ax2.get_yticks()
#ax2.set_yticklabels(['{:.0%}'.format(x) for x in vals])
plot.minorticks_on()
plot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plot.grid()

csv=np.genfromtxt('log_leaky_relu', delimiter=':')
plot.plot(1-csv[:,1], label="leaky_relu error")

csv=np.genfromtxt('log_relu', delimiter=':')
plot.plot(1-csv[:,1], label="relu error")

csv=np.genfromtxt('log_selu', delimiter=':')
plot.plot(1-csv[:,1], label="selu error")

csv=np.genfromtxt('log_tanh', delimiter=':')
plot.plot(1-csv[:,1], label="tanh error")

csv=np.genfromtxt('log_sigmoid', delimiter=':')
plot.plot(1-csv[:,1], label="sigmoid error")

plot.legend(loc='upper right')
plot.show()

