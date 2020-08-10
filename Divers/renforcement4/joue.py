import gym
import cv2
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import time
import matplotlib.pyplot as plot
import os

env=gym.make("MsPacman-v0")

#model=tf.keras.models.load_model('my_model_v1')
model=tf.keras.models.load_model('my_model_target')

decalage_debut=90
taille_sequence=6

def transform_img(image):
  result=np.expand_dims(image[:170, :, 0], axis=-1)
  return result

def joue():

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

  tab_img=[]
  score=0
  vie=3
  while True:
    valeurs_q=model(np.expand_dims(np.concatenate(tab_sequence, axis=-1), axis=0))
    action=int(tf.argmax(valeurs_q[0], axis=-1))
    print(action+1, end=' ')
    score+=min(reward, 10.)
    if info['ale.lives']<vie:
      print("XXX", end=" ")
      vie-=1
    if done:
      print("\nSCORE:", score)
      return tab_img, score
    
    observation, reward, done, info=env.step(action+1)
    if reward>10:
      print("MIAM", reward, end=" ")

    img=transform_img(observation)
    tab_sequence[:-1]=tab_sequence[1:]
    tab_sequence[taille_sequence-1]=img
    tab_img.append(observation)
    
score=0
while score<1400:
  start_time=time.time()
  tab_img, score=joue()
  print(time.time()-start_time)
      
for i in range(len(tab_img)):
  cv2.imshow("Pacman", tab_img[i])
  key=cv2.waitKey(20)
  if key==ord('q'):
    break
    
