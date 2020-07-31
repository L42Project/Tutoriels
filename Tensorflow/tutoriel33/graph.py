import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv
import sys
import numpy as np

if len(sys.argv)!=2:
    print("Usage:", sys.argv[0], "<fichier csv>")
    quit()
fichier=sys.argv[1]

def calc(tab_data, fenetre):
    tab_m=[]
    for i in range(len(tab_data)-fenetre):
        m=np.mean(tab_data[i:i+fenetre])
        tab_m.append(m)
    return tab_m

x=[]
accuracy=[]
loss=[]
val_accuracy=[]
val_loss=[]
fenetre=50

val=0
with open(fichier,'r') as csvfile:
    plots=csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        x.append(float(row[0]))
        accuracy.append(float(row[1]))
        loss.append(float(row[2]))
        if len(row)==5:
            val_accuracy.append(float(row[3]))
            val_loss.append(float(row[4]))
            val=1


fig, (ax1, ax2)=plt.subplots(2)
fig.set_size_inches(9, 7, forward=True)

ax1.set_ylim([0, 1.0])
ax1.grid(which='both')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ln=ax1.plot(x, accuracy, label='Accuracy')

ax1_=ax1.twinx()
ax1_.set_ylim([0.0, 2.0])
ln_=ax1_.plot(x, loss, label='Loss', color='red')

lns=ln+ln_
labs=[l.get_label() for l in lns]

ax2.set_ylim([0, 1.0])
ax2.grid(which='both')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ln=ax2.plot(x, val_accuracy, label='Val accuracy')

ax2_=ax2.twinx()
ax2_.set_ylim([0.0, 2.0])
ln_=ax2_.plot(x, val_loss, label='Val loss', color='red')

lns=ln+ln_
labs=[l.get_label() for l in lns]
ax2.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)

plt.show()
