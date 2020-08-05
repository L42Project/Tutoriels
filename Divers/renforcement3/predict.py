import gym
import tensorflow as tf
import numpy as np

env=gym.make("CartPole-v0")
env._max_episode_steps=500

model=tf.keras.models.load_model("my_model")

while True:
  observations=env.reset()
  score=0
  while True:
    env.render()
    valeurs_q=model(np.expand_dims(observations, axis=0))
    action=int(tf.argmax(valeurs_q[0], axis=-1))
    observations, reward, done, info=env.step(action)
    if done:
      print("SCORE", score)
      break
    score+=1
