import gym
import numpy as np
import CartPole_common

env=gym.make("CartPole-v0")
env._max_episode_steps=5000

q_table=np.load("CartPole_qtable.npy")

for epoch in range(1000):
    state = env.reset()
    score = 0
    while True:
        env.render()
        discrete_state=CartPole_common.discretise(state)
        action=np.argmax(q_table[discrete_state])
        #if not np.random.randint(5):
        #    action=np.random.randint(2)
        state, reward, done, info=env.step(action)
        score+=reward
        if done:
            print('Essai {:05d} Score: {:04d}'.format(epoch, int(score)))
            break

env.close()
