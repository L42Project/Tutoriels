import tensorflow as tf
import time, threading
import numpy as np
import cv2
import model_cond

noise_dim=100

generator=model_cond.generator_model()
checkpoint=tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir='./training_checkpoints_gan_cond/'))

marge=20
marge2=5
image=np.zeros(shape=(28+2*marge, 6*28+2*marge+4*marge2), dtype=np.float32)
old_h1=old_h2=old_m1=old_m2=old_s1=old_s2=-1
cont=1

def foo():
    global old_h1, old_h2, old_m1, old_m2, old_s1, old_s2

    if cont:
        threading.Timer(1, foo).start()
    seed=tf.random.normal([6, noise_dim])
    heure=time.strftime('%H:%M:%S')
    print(heure)
    h1=int(int(heure.split(':')[0])/10)
    h2=int(heure.split(':')[0])%10
    m1=int(int(heure.split(':')[1])/10)
    m2=int(heure.split(':')[1])%10
    s1=int(int(heure.split(':')[2])/10)
    s2=int(heure.split(':')[2])%10
    labels=tf.one_hot([h1, h2, m1, m2, s1, s2], 10)

    prediction=generator([seed, labels], training=False)
    if h1!=old_h1:
        image[0+marge:28+marge, 0*28+marge:1*28+marge]=prediction[0, :, :, 0]*127.5+127.5
    if h2!=old_h2:
        image[0+marge:28+marge, 1*28+marge:2*28+marge]=prediction[1, :, :, 0]*127.5+127.5
    if m1!=old_m1:
        image[0+marge:28+marge, 2*28+marge+2*marge2:3*28+marge+2*marge2]=prediction[2, :, :, 0]*127.5+127.5
    if m2!=old_m2:
        image[0+marge:28+marge, 3*28+marge+2*marge2:4*28+marge+2*marge2]=prediction[3, :, :, 0]*127.5+127.5
    if s1!=old_s1:
        image[0+marge:28+marge, 4*28+marge+4*marge2:5*28+marge+4*marge2]=prediction[4, :, :, 0]*127.5+127.5
    if s2!=old_s2:
        image[0+marge:28+marge, 5*28+marge+4*marge2:6*28+marge+4*marge2]=prediction[5, :, :, 0]*127.5+127.5

    cv2.circle(image, (marge+2*28+marge2, marge+8), 1, (255, 255, 255), 2)
    cv2.circle(image, (marge+2*28+marge2, marge+20), 1, (255, 255, 255), 2)
    cv2.circle(image, (marge+4*28+3*marge2, marge+8), 1, (255, 255, 255), 2)
    cv2.circle(image, (marge+4*28+3*marge2, marge+20), 1, (255, 255, 255), 2)

    old_h1=h1
    old_h2=h2
    old_m1=m1
    old_m2=m2
    old_s1=s1
    old_s2=s2

foo()
while True:
    cv2.imshow("Horloge", image.astype(np.uint8))
    key=cv2.waitKey(10)
    if key==ord('q')&0xFF:
        cv2.destroyAllWindows()
        cont=0
        quit()
