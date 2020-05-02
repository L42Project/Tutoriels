import numpy as np
import time

Q=[[0, 0], [0, 0], [0, 0]]

T=[[[0.50, 0.00, 0.50], [0.00, 0.00, 1.00]],
   [[0.70, 0.10, 0.20], [0.00, 0.95, 0.05]],
   [[0.40, 0.00, 0.60], [0.30, 0.30, 0.40]]]

R=[[[ 0.00,  0.00, 0.00], [ 0.00,  0.00,  0.00]],
   [[+5.00,  0.00, 0.00], [ 0.00,  0.00,  0.00]],
   [[ 0.00,  0.00, 0.00], [-1.00,  0.00,  0.00]]]

gamma=0.95

for i in range(200):
    time.sleep(0.05)
    tab_somme_action=[]
    for S in range(3):
        for A in range(2):
            somme=0
            for s in range(3):
                somme+=T[S][A][s]*(R[S][A][s]+gamma*np.max(Q[s]))
            Q[S][A]=somme

    print("---------------------------------")
    print("Iteration:", i)
    for S in range(3):
        print()
        for A in range(2):
            text="Q[etat:{}, action:{}]={:+10.4f}".format(S, A, Q[S][A])
            if A==np.argmax(Q[S]):
                text=text+" <-"
            print(text)
print("---------------------------------")
