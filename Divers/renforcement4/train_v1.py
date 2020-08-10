import gym
import cv2
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import time
import matplotlib.pyplot as plot

env = gym.make("MsPacman-v0")
print("Liste des actions", env.unwrapped.get_action_meanings())
nbr_action=tf.constant(4)

file_model='my_model_v1'
file_stats='tab_score_v1'

gamma=tf.constant(0.999)
epoch=1500
decalage_debut=90
taille_sequence=6
nbr_jeu=40
pourcentage_batch=0.20
best_score=0

epsilon=1.
epsilon_min=0.10
start_epsilon=1
end_epsilon=epoch//4
epsilon_decay_value=epsilon/(end_epsilon-start_epsilon)

def model(nbr_cc=8):
  entree=layers.Input(shape=(170, 160, taille_sequence), dtype='float32')
  result=layers.Conv2D(  nbr_cc, 3, activation='relu', padding='same', strides=2)((entree/128)-1)
  result=layers.Conv2D(2*nbr_cc, 3, activation='relu', padding='same', strides=2)(result)
  result=layers.BatchNormalization()(result)
  result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same', strides=2)(result)
  result=layers.Conv2D(8*nbr_cc, 3, activation='relu', padding='same', strides=2)(result)
  result=layers.BatchNormalization()(result)

  result=layers.Flatten()(result)

  result=layers.Dense(512, activation='relu')(result)
  sortie=layers.Dense(nbr_action)(result)
    
  model=models.Model(inputs=entree, outputs=sortie)
  return model

def transform_img(image):
  result=np.expand_dims(image[:170, :, 0], axis=-1)
  return result

def simulation(epsilon, debug=False):
  if debug:
    start_time=time.time()

  tab_observations=[]
  tab_rewards=[]
  tab_actions=[]
  tab_next_observations=[]
  tab_done=[]
  
  ######
  observations=env.reset()
  vie=3
  for i in range(decalage_debut-taille_sequence):
    env.step(0)
  tab_sequence=[]
  for i in range(taille_sequence):
    observation, reward, done, info=env.step(0)
    img=transform_img(observation)
    tab_sequence.append(img)
  tab_sequence=np.array(tab_sequence, dtype=np.float32)
  ######

  score=0
  while True:
    if np.random.random()>epsilon:
      valeurs_q=model(np.expand_dims(np.concatenate(tab_sequence, axis=-1), axis=0))
      action=int(tf.argmax(valeurs_q[0], axis=-1))
    else:
      action=np.random.randint(0, nbr_action)

    h=np.random.randint(10)
    if h==0:
      tab_observations.append(np.concatenate(tab_sequence, axis=-1))
      tab_actions.append(action)
    score+=reward
    if info['ale.lives']<vie:
      reward=-50.
      vie=info['ale.lives']
      if h==0:
        tab_done.append(True)
    else:
      if h==0:
        tab_done.append(done)
    if h==0:
      tab_rewards.append(reward)
    if done:
      tab_s.append(score)
      if h==0:
        tab_sequence[:-1]=tab_sequence[1:]
        tab_sequence[taille_sequence-1]=img
        tab_next_observations.append(np.concatenate(tab_sequence, axis=-1))
      tab_done=np.array(tab_done, dtype=np.float32)
      tab_observations=np.array(tab_observations, dtype=np.float32)
      tab_next_observations=np.array(tab_next_observations, dtype=np.float32)
      tab_rewards=np.array(tab_rewards, dtype=np.float32)
      tab_rewards[tab_rewards==0]=-1.
      tab_rewards[tab_rewards>10]=10.
      tab_actions=np.array(tab_actions, dtype=np.int32)
      if debug:
            print("  Creation observations {:5.3f} seconde(s)".format(float(time.time()-start_time)))
            print("     score:{:5d}   batch:{:4d}".format(int(score), len(tab_done)))
      return tab_observations,\
             tab_rewards,\
             tab_actions,\
             tab_next_observations,\
             tab_done
    observation, reward, done, info=env.step(action+1)
    img=transform_img(observation)
    tab_sequence[:-1]=tab_sequence[1:]
    tab_sequence[taille_sequence-1]=img
    if h==0:
      tab_next_observations.append(np.concatenate(tab_sequence, axis=-1))

def my_loss(y, q):
  loss=tf.reduce_mean(tf.math.square(y-q))
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
  global epsilon, best_score
  for e in range(epoch):
    for i in range(nbr_jeu):
      print("Epoch {:04d}/{:05d} epsilon={:05.3f}".format(i, e, epsilon))
      tab_observations, tab_rewards, tab_actions, tab_next_observations, tab_done=simulation(epsilon, debug=True)
      if debug:
        start_time=time.time()
      train_step(tab_rewards, tab_actions, tab_observations, tab_next_observations, tab_done)
      if debug:
        print("  Entrainement {:5.3f} seconde(s)".format(float(time.time()-start_time)))
        print("     loss: {:6.4f}".format(train_loss.result()))
      train_loss.reset_states()

    epsilon-=epsilon_decay_value
    epsilon=max(epsilon, epsilon_min)
    np.save(file_stats, tab_s)
    if np.mean(tab_s[-200:])>best_score:
      print("Sauvegarde du modele")
      model.save(file_model)
      best_score=np.mean(tab_s[-200:])

model=model(16)

optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
train_loss=tf.keras.metrics.Mean()
tab_s=[]
train(debug=True)
