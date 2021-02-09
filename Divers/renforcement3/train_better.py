import gym
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

env=gym.make("CartPole-v0")
env._max_episode_steps=200
nbr_action=2

fichier_log=open("log_critic.csv", "a")

gamma=0.98
max_episode=600
epsilon=1.
epsilon_min=0.10
start_epsilon=10
end_epsilon=max_episode
epsilon_decay_value=epsilon/(end_epsilon-start_epsilon)

def model():
  entree=layers.Input(shape=(4), dtype='float32')
  result=layers.Dense(32, activation='relu')(entree)
  result=layers.Dense(32, activation='relu')(result)
  sortie=layers.Dense(nbr_action)(result)

  model=models.Model(inputs=entree, outputs=sortie)
  return model

def my_loss(target_q, predicted_q):
  loss=tf.reduce_mean(tf.math.square(target_q-predicted_q))
  return loss

@tf.function
def train_step(reward, action, observation, next_observation, done):
  next_Q_values=model(next_observation)
  best_next_actions=tf.math.argmax(next_Q_values, axis=1)
  next_mask=tf.one_hot(best_next_actions, nbr_action)
  next_best_Q_values=tf.reduce_sum(next_Q_values*next_mask, axis=1)
  target_Q_values=reward+(1-done)*gamma*next_best_Q_values
  target_Q_values=tf.reshape(target_Q_values, (-1, 1))
  mask=tf.one_hot(action, nbr_action)
  with tf.GradientTape() as tape:
    all_Q_values=model(observation)
    Q_values=tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
    loss=my_loss(target_Q_values, Q_values)
  gradients=tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)

def train(debug=False):
  global epsilon
  m_reward=0
  for episode in range(max_episode):
    score=0
    tab_observations=[]
    tab_rewards=[]
    tab_actions=[]
    tab_next_observations=[]
    tab_done=[]

    observations=env.reset()
    score=0
    while True:
      tab_observations.append(observations)
      if np.random.random()>epsilon:
        valeurs_q=model(np.expand_dims(observations, axis=0))
        action=int(tf.argmax(valeurs_q[0], axis=-1))
      else:
        action=np.random.randint(0, nbr_action)
      observations, reward, done, info=env.step(action)
      score+=reward
      tab_actions.append(action)
      tab_next_observations.append(observations)
      tab_done.append(done)
      if done:
        tab_rewards.append(-10.)
        break
      tab_rewards.append(reward)

    tab_rewards=np.array(tab_rewards, dtype=np.float32)
    tab_actions=np.array(tab_actions, dtype=np.int32)
    tab_observations=np.array(tab_observations, dtype=np.float32)
    tab_next_observations=np.array(tab_next_observations, dtype=np.float32)
    tab_done=np.array(tab_done, dtype=np.float32)
    train_step(tab_rewards, tab_actions, tab_observations, tab_next_observations, tab_done)
    train_loss.reset_states()

    epsilon-=epsilon_decay_value
    epsilon=max(epsilon, epsilon_min)

    m_reward=0.05*score+(1-0.05)*m_reward
    message="Episode {:04d}  score:{:6.1f}  moyenne lissée: {:6.1f} (epsilon={:5.3f})"
    print(message.format(episode, score, m_reward, epsilon))

    fichier_log.write("{:f}:{:f}\n".format(score, m_reward))

    if m_reward>env._max_episode_steps-10:
        print("Fin de l'apprentissage".format(episode))
        break

model=model()
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-2)
train_loss=tf.keras.metrics.Mean()
tab_s=[]

train()

fichier_log.close()
