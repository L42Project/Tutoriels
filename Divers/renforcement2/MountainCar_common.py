import numpy as np

# Valeurs hautes et basses des observations
low_values =np.array([-1.2, -0.07])
high_values=np.array([0.6, 0.07])

division=[42, 42]
pas=(high_values-low_values)/division

def discretise(state):
    discrete_state=(state-low_values)/pas
    return tuple(discrete_state.astype(np.int))
