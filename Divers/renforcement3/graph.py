import numpy as np
import matplotlib.pyplot as plot
import sys

fenetre=50
max_score=500

if len(sys.argv)!=2:
    print("Usage:", sys.argv[0], "<fichier npy>")
    quit()

tab_s=np.load(sys.argv[1])

tab_m=[]
for i in range(len(tab_s)-fenetre):
    m=np.mean(tab_s[i:i+fenetre])
    tab_m.append(m)

fig=plot.gcf()
fig.set_size_inches(12, 6)
plot.plot(tab_s)
plot.grid()
plot.ylim(0, max_score)
plot.plot(np.arange(fenetre, len(tab_s)), tab_m, color='#FF0000')
plot.show()
