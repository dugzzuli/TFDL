import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_data = tf.placeholder("float32", [None, 784])
y_data = tf.placeholder("float32", [None, 10])

weight1 = tf.Variable(tf.ones([784, 256]))
bias1 = tf.Variable(tf.ones([256]))
y_model1 = tf.nn.relu(tf.matmul(x_data, weight1) + bias1)

weight2 = tf.Variable(tf.ones([256, 10]))
bias2 = tf.Variable(tf.ones([10]))
y_model = tf.nn.relu(tf.matmul(y_model1, weight2) + bias2)


loss = tf.reduce_mean(-tf.reduce_sum(y_data*tf.log(tf.nn.softmax(y_model)), reduction_indices=[1]))
#loss = -tf.reduce_sum(y_data*tf.log(y_model), reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x_data: batch_xs, y_data: batch_ys})
    if _ % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x_data: mnist.test.images,
                                            y_data: mnist.test.labels}))

