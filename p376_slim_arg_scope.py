import tensorflow.contrib.slim as slim
import tensorflow as tf


with slim.arg_scope([slim.comv2d],
                    padding="SAME",
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='coonv2')
    net = slim.conv2d(net, 256, [11, 11], scope='conv3')
