import numpy as np
import matplotlib.pyplot as plot
from L42Project import ia as LPia
import sys

fig, ax = plot.subplots()

plot.ylim(0, 0.40)
plot.minorticks_on()
plot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plot.grid()

csv=np.genfromtxt('log_sd', delimiter=':')
plot.plot(1-csv[:,2], label="Erreur sans dropout")

y_sd=min(1-csv[:,2])
x_sd=int(csv[np.where((1-csv[:,2])==y_sd)][0][0])

ax.annotate('Minumum: {:f}'.format(y_sd), xy=(x_sd, y_sd), xytext=(70, 0.3),
            arrowprops=dict(facecolor='black', shrink=0.05),
)

csv=np.genfromtxt('log_ad', delimiter=':')
plot.plot(1-csv[:,2], label="Erreur avec dropout")

y_ad=min(1-csv[:,2])
x_ad=int(csv[np.where((1-csv[:,2])==y_ad)][0][0])

ax.annotate('Minumum: {:f}'.format(y_ad), xy=(x_ad, y_ad), xytext=(80, 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05),
)

plot.legend(loc='upper right')
plot.show()

