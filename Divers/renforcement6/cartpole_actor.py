import gym
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import os

env=gym.make("CartPole-v0")
env._max_episode_steps=200
nbr_actions=2
gamma=0.99
max_episode=600

prefix_log_file="log_actor"
id_file=0
while os.path.exists(prefix_log_file+str(id_file)+".csv"):
    id_file+=1
fichier_log=open(prefix_log_file+str(id_file)+".csv", "w")
print("Création du fichier de log", prefix_log_file+str(id_file)+".csv")

def model(nbr_inputs, nbr_hidden, nbr_actions):
  entree=layers.Input(shape=(nbr_inputs), dtype='float32')
  result=layers.Dense(32, activation='relu')(entree)
  result=layers.Dense(32, activation='relu')(result)
  sortie=layers.Dense(nbr_actions, activation='softmax')(result)

  my_model=models.Model(inputs=entree, outputs=sortie)
  return my_model

def calcul_discount_rate(rewards_history, gamma, normalize=False):
    result=[]
    discounted_sum=0
    for r in rewards_history[::-1]:
        discounted_sum=r+gamma*discounted_sum
        result.insert(0, discounted_sum)

    # Normalisation
    if normalize is True:
        result=np.array(result)
        result=(result-np.mean(result))/(np.std(result)+1E-7)
        result=list(result)

    return result

def train():
  m_reward=0
  for episode in range(max_episode):
    tab_rewards=[]
    tab_prob_actions=[]

    observations=env.reset()
    with tf.GradientTape() as tape:
        while True:
                action_probs=my_model(np.expand_dims(observations, axis=0))
                action=np.random.choice(nbr_actions, p=np.squeeze(action_probs))
                tab_prob_actions.append(action_probs[0, action])
                observations, reward, done, info=env.step(action)
                tab_rewards.append(reward)
                if done:
                    break

        discount_rate=calcul_discount_rate(tab_rewards, gamma, normalize=True)

        loss=-tf.math.log(tab_prob_actions)*discount_rate
        gradients=tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))

    score=sum(tab_rewards)
    m_reward=0.05*score+(1-0.05)*m_reward
    message="Episode {:04d}  score:{:6.1f}  MPE: {:6.1f}"
    print(message.format(episode, score, m_reward))

    fichier_log.write("{:f}:{:f}\n".format(score, m_reward))

    if m_reward>env._max_episode_steps-10:
        print("Fin de l'apprentissage".format(episode))
        break

my_model=model(4, 32, nbr_actions)
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-2)

train()

fichier_log.close()
