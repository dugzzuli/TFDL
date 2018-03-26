import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

threshold = 1.0e-2
x1_data = np.random.randn(100).astype(np.float32)
x2_data = np.random.randn(100).astype(np.float32)
y_data = x1_data * 2 + x2_data * 3 + 1.5

weight1 = tf.Variable(1.0)
weight2 = tf.Variable(1.0)
bias = tf.Variable(1.0)
x1_ = tf.placeholder(tf.float32)
x2_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

y_model = tf.add(tf.add(tf.multiply(x1_, weight1), tf.multiply(x2_, weight2)), bias)
loss = tf.reduce_mean(tf.pow((y_model - y_), 2))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
flag = 1
while flag:
    for (x, y) in zip(zip(x1_data, x2_data), y_data):
        sess.run(train_op, feed_dict = {x1_: x[0], x2_: x[1], y_: y})
    if sess.run(loss, feed_dict={x1_: x[0], x2_: x[1], y_: y}) <= threshold:
        flag = 0

fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(x1_data, x2_data)
Z = sess.run(weight1) * X + sess.run(weight2) * Y + sess.run(bias)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
ax.contourf(X, Y, Z, zdir='z', offset=-1, cmap=plt.cm.hot)
ax.set_zlim(-1, 1)
plt.show()