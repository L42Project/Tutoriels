import numpy as np

# Valeurs hautes et basses des observations
low_values=np.array([-5, -5, -0.45, -5])
high_values=np.array([5, 5, 0.45, 5])

division=[42, 42, 42, 42]
pas=(high_values-low_values)/division

def discretise(state):
    discrete_state=(state-low_values)/pas
    return tuple(discrete_state.astype(np.int))
