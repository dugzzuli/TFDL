import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input = tf.Variable(tf.random_normal([1, 3, 3, 1]))
filter = tf.Variable(tf.ones([1, 1, 1, 1]))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    cov2d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="VALID")
    print(sess.run(cov2d))

