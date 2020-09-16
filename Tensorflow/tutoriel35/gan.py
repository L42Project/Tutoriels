import tensorflow as tf
import glob
import numpy as np
import os
from tensorflow.keras import layers, models
import time
import cv2
import model

batch_size=256
epochs=500
noise_dim=100
tab_size=5
num_examples_to_generate=tab_size*tab_size
dir_images='images_gan'
checkpoint_dir='./training_checkpoints_gan'
checkpoint_prefix=os.path.join(checkpoint_dir, "ckpt")

if not os.path.isdir(dir_images):
    os.mkdir(dir_images)

(train_images, train_labels), (test_images, test_labels)=tf.keras.datasets.mnist.load_data()

train_images=train_images.reshape(-1, 28, 28, 1).astype('float32')
train_images=(train_images-127.5)/127.5

train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(batch_size)

def discriminator_loss(real_output, fake_output):
    real_loss=cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss=cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss=real_loss+fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator=model.generator_model()
discriminator=model.discriminator_model()

cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)

train_generator_loss=tf.keras.metrics.Mean()
train_discriminator_loss=tf.keras.metrics.Mean()

generator_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

checkpoint=tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed=tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise=tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images=generator(noise, training=True)

      real_output=discriminator(images, training=True)
      fake_output=discriminator(generated_images, training=True)

      gen_loss=generator_loss(fake_output)
      disc_loss=discriminator_loss(real_output, fake_output)

      train_generator_loss(gen_loss)
      train_discriminator_loss(disc_loss)
      
    gradients_of_generator=gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator=disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start=time.time()
    for image_batch in dataset:
      train_step(image_batch)
    generate_and_save_images(generator, epoch+1, seed)
    if (epoch+1)%15==0:
      checkpoint.save(file_prefix=checkpoint_prefix)
    print ('Epoch {}: loss generator: {:.4f} loss discriminator: {:.4f}  {:.4f} sec'.format(epoch+1,
                                                                                            train_generator_loss.result(),
                                                                                            train_discriminator_loss.result(),
                                                                                            time.time()-start))
    train_generator_loss.reset_states()
    train_discriminator_loss.reset_states()
    
def generate_and_save_images(model, epoch, test_input):
  labels=tf.one_hot(tf.range(0, num_examples_to_generate, 1)%10, 10)
  predictions=model([test_input, labels], training=False)
  img=np.empty(shape=(tab_size*28, tab_size*28), dtype=np.float32)
  for i in range(tab_size):
    for j in range(tab_size):
      img[j*28:(j+1)*28, i*28:(i+1)*28]=predictions[j*tab_size+i, :, :, 0]*127.5+127.5
  cv2.imwrite('{}/image_{:04d}.png'.format(dir_images, epoch), img)
  
train(train_dataset, epochs)
