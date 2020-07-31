import numpy as np
import matplotlib.pyplot as plot
from L42Project import ia as LPia
import sys

fig, ax = plot.subplots()

plot.ylim(0, 1)
plot.minorticks_on()
plot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plot.grid()

csv=np.genfromtxt('log_1', delimiter=':')
plot.plot(1-csv[:,2], label="Erreur sans dropout")

y_sd=min(1-csv[:,2])
x_sd=int(csv[np.where((1-csv[:,2])==y_sd)][0][0])

ax.annotate('Minumum: {:f}'.format(y_sd), xy=(x_sd, y_sd), xytext=(70, 0.27),
            arrowprops=dict(facecolor='black', shrink=0.05),
)

csv=np.genfromtxt('log_2', delimiter=':')
plot.plot(1-csv[:,2], label="Erreur avec dropout")

y_ad=min(1-csv[:,2])
x_ad=int(csv[np.where((1-csv[:,2])==y_ad)][0][0])

ax.annotate('Minumum: {:f}'.format(y_ad), xy=(x_ad, y_ad), xytext=(80, 0.08),
            arrowprops=dict(facecolor='black', shrink=0.05),
)

csv=np.genfromtxt('log_3', delimiter=':')
plot.plot(1-csv[:,2], label="Erreur méthode complétion dataset")

y_ad=min(1-csv[:,2])
x_ad=int(csv[np.where((1-csv[:,2])==y_ad)][0][0])

ax.annotate('Minumum: {:f}'.format(y_ad), xy=(x_ad, y_ad), xytext=(20, 0.03),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plot.legend(loc='upper right')
plot.show()

