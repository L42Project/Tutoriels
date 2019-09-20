import tensorflow as tf

x = tf.Variable(4.0)
with tf.GradientTape() as g:
  with tf.GradientTape() as gg:
    y=x*x
  dy_dx=gg.gradient(y, x)     # Will compute to 6.0
d2y_dx2=g.gradient(dy_dx, x)  # Will compute to 2.0

tf.print(dy_dx)
tf.print(d2y_dx2)
