import gym
import numpy as np
import MountainCar_common

env=gym.make("MountainCar-v0")

# Coefficient d'apprentissage
alpha=0.1
# Le "discount rate"
gamma=0.98

epoch=25000
show_every=500

# Politique exploration/exploitation
epsilon=1.
epsilon_min=0.1
start_epsilon=1
end_epsilon=epoch//2
epsilon_decay_value=epsilon/(end_epsilon-start_epsilon)

nbr_action=env.action_space.n
q_table=np.random.uniform(low=-1, high=1, size=(MountainCar_common.division+[nbr_action]))

OK=0
for episode in range(epoch):
    obs=env.reset()
    discrete_state=MountainCar_common.discretise(obs)
    done=False

    if episode%show_every == 0:
        render=True
        print("epoch {:06d}/{:06d}  reussite:{:04d}/{:04d}  epsilon={:08.6f}".format(episode, epoch, OK, show_every, epsilon))
        OK=0
    else:
        render=False

    while not done:

        if np.random.random()>epsilon:
            action=np.argmax(q_table[discrete_state])
        else:
            action=np.random.randint(nbr_action)

        new_state, reward, done, info=env.step(action)
        new_discrete_state=MountainCar_common.discretise(new_state)
        if episode%show_every == 0:
            env.render()

        if new_state[0]>=env.goal_position:
            reward=1
            OK+=1

        # Mise à jour de Q(s, a) avec la formule de Bellman
        max_future_q=np.max(q_table[new_discrete_state])
        current_q=q_table[discrete_state][action]
        new_q=(1-alpha)*current_q+alpha*(reward+gamma*max_future_q)
        q_table[discrete_state][action]=new_q

        discrete_state=new_discrete_state

    if end_epsilon>=episode>=start_epsilon:
        epsilon-=epsilon_decay_value
        if epsilon<epsilon_min:
            epsilon=epsilon_min

np.save("MountainCar_qtable", q_table)
env.close()
