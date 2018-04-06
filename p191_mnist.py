import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float32", [None, 784])
W = tf.Variable(tf.ones([784, 10]))
b = tf.Variable(tf.ones([10]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.nn.softmax(y)), reduction_indices=[1]))

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#loss = tf.reduce_sum(tf.pow((y - y_), 2))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#sess = tf.Session()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#init = tf.global_variables_initializer()
#sess.run(init)

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

