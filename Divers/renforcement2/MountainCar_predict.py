import gym
import numpy as np
import MountainCar_common

env=gym.make("MountainCar-v0")

q_table=np.load("MountainCar_qtable.npy")

for epoch in range(1000):
    state = env.reset()
    while True:
        env.render()
        discrete_state=MountainCar_common.discretise(state)
        action=np.argmax(q_table[discrete_state])
        state, reward, done, info=env.step(action)
        if done:
            print("Essai {:05d}: {}".format(epoch, "OK" if state[0]>=env.goal_position else "raté ..."))
            break
env.close()
