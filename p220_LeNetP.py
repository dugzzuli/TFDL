import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

#训练数据
x = tf.placeholder('float', [None, 784])
#训练标签数据
y_ = tf.placeholder('float', [None, 10])
#把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽，第4代表图像通道数，1表示黑白
x_image = tf.reshape(x, [-1, 28, 28, 1])

#第一层：卷积层
#filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
filter1 = tf.get_variable("filter1", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
#bias1 = tf.Variable(tf.truncated_normal([32]))
bias1 = tf.get_variable("bias1", [32], initializer=tf.constant_initializer(0.0))
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding="SAME")
#h_conv1 = tf.nn.sigmoid(conv1 + bias1)
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))

# 第二层：最大池化层
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 第三层：卷积层
#filter2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
filter2 = tf.get_variable("filter2", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
#bias2 = tf.Variable(tf.truncated_normal([64]))
bias2 = tf.get_variable("bias2", [64], initializer=tf.constant_initializer(0.0))
conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding='SAME')
#h_conv2 = tf.nn.sigmoid(conv2 + bias2)
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))

# 第四层：最大池化层
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第五层：全连接层
#W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
W_fc1 = tf.get_variable('W_fc1', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
#b_fc1 = tf.Variable(tf.truncated_normal([1024]))
b_fc1 = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))
h_pool2_flat = tf.reshape(maxPool3, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，加入Dropout层
keep_prob = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# 第六层： 全连接层 ——最后一个
#W_fc2 = tf.Variable(tf.truncated_normal([1024, 10]))
W_fc2 = tf.get_variable("W_fc2", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
#b_fc2 = tf.Variable(tf.truncated_normal([10]))
b_fc2 = tf.get_variable("b_fc2", [10], initializer=tf.constant_initializer(0.1))
fc2 = tf.matmul(fc1_dropout, W_fc2) + b_fc2

# 第七层：输出层
y_conv = tf.nn.softmax(fc2)
#
# 定义交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
#train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


# tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值
# 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#用亚运会来统计测试准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
sess.run(tf.global_variables_initializer())
#start_time = time.time()
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        #end_time = time.time()
        #print("time: ", (end_time - start_time))
        print(
            "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        print("validation accuracy %g" % accuracy.eval(
            feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

sess.close()

