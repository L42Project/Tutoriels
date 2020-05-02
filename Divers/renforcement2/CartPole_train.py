import gym
import numpy as np
import cv2
import CartPole_common

env=gym.make("CartPole-v0")
env._max_episode_steps=500

alpha=0.05
gamma=0.98

epoch=50000
show_every=500

epsilon=1.
epsilon_min=0.05
start_epsilon=1
end_epsilon=epoch//2
epsilon_decay_value=epsilon/(end_epsilon-start_epsilon)

nbr_action=env.action_space.n
q_table=np.random.uniform(low=-1, high=1, size=(CartPole_common.division+[nbr_action]))

result_done=0
scores=[]
best_score=0
for episode in range(epoch):
    obs=env.reset()
    discrete_state=CartPole_common.discretise(obs)
    done=False

    if episode%show_every == 0:
        render=True
        mean_score=np.mean(scores)
        print("Epoch {:06d}/{:06d} reussite:{:04d}/{:04d} epsilon={:06.4f} Mean score={:08.4f} alpha={:06.4f}".format(episode, epoch, result_done, show_every, epsilon, mean_score, alpha))
        scores=[]
        result_done=0
        if mean_score>best_score:
            print("Sauvegarde ...")
            np.save("CartPole_qtable", q_table)
            best_score=mean_score
        alpha=alpha*0.99

    else:
        render=False

    score=1
    while not done:

        if np.random.random()>epsilon:
            action=np.argmax(q_table[discrete_state])
        else:
            action=np.random.randint(nbr_action)

        new_state, reward, done, info=env.step(action)
        new_discrete_state=CartPole_common.discretise(new_state)

        if episode%show_every == 0:
            env.render()

        #reward=2-np.abs(new_state[0])
        if done:
            scores.append(score)
            if score==env._max_episode_steps:
                result_done+=1
            else:
                reward=-10

        max_future_q=np.max(q_table[new_discrete_state])
        current_q=q_table[discrete_state][action]
        new_q=(1-alpha)*current_q+alpha*(reward+gamma*max_future_q)
        q_table[discrete_state][action]=new_q

        score+=1
        discrete_state=new_discrete_state

    if end_epsilon>=episode>=start_epsilon:
        epsilon-=epsilon_decay_value
        if epsilon<epsilon_min:
            epsilon=epsilon_min

env.close()
