import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

gamma=0.99
max_steps_per_episode=10000
env=gym.make("CartPole-v0")
env._max_episode_steps=200

prefix_log_file="log_actor_critic_dsum_"
id_file=0
while os.path.exists(prefix_log_file+str(id_file)+".csv"):
    id_file+=1
fichier_log=open(prefix_log_file+str(id_file)+".csv", "w")
print("Création du fichier de log", prefix_log_file+str(id_file)+".csv")

nbr_actions=2
nbr_inputs=4

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

def my_model(nbr_inputs, nbr_hidden, nbr_actions):
    entree=layers.Input(shape=(nbr_inputs), dtype='float32')

    common=layers.Dense(nbr_hidden, activation="relu")(entree)
    action=layers.Dense(nbr_actions, activation="softmax")(common)
    critic=layers.Dense(1)(common)

    model=keras.Model(inputs=entree, outputs=[action, critic])
    return model

model=my_model(nbr_inputs, 32, nbr_actions)

optimizer=keras.optimizers.Adam(learning_rate=1E-2)
huber_loss=keras.losses.Huber()

m_reward=0
episode=0

while True:
    action_probs_history=[]
    critic_value_history=[]
    rewards_history=[]

    state=env.reset()
    episode_reward=0
    with tf.GradientTape() as tape:

        # Récupération de données
        for timestep in range(1, max_steps_per_episode):
            action_probs, critic_value=model(np.expand_dims(state, axis=0))
            critic_value_history.append(critic_value[0, 0])
            action=np.random.choice(nbr_actions, p=np.squeeze(action_probs))
            action_probs_history.append(action_probs[0, action])
            state, reward, done, infos=env.step(action)
            rewards_history.append(reward)
            episode_reward+=reward
            if done:
                break

        discount_rate=calcul_discount_rate(rewards_history, gamma, normalize=True)

        history=zip(action_probs_history, critic_value_history, discount_rate)
        actor_losses=[]
        critic_losses=[]
        for action_prob, critic_value, discount_rate in history:
            actor_losses.append(-tf.math.log(action_prob)*(discount_rate-critic_value))
            critic_losses.append(huber_loss([critic_value], [discount_rate]))

        loss_value=tf.reduce_mean(actor_losses+critic_losses)
        grads=tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode+=1
    m_reward=0.05*episode_reward+(1-0.05)*m_reward

    message="Episode {:04d}  score:{:6.1f}  MPE: {:6.1f}"
    print(message.format(episode, episode_reward, m_reward))

    fichier_log.write("{:f}:{:f}\n".format(episode_reward, m_reward))

    if m_reward>env._max_episode_steps-10:
        print("Fin de l'apprentissage".format(episode))
        break

fichier_log.close()
model.save("my_model")
