import tensorflow as tf


def conv(layer_name, x, out_channels, kernel_size, stride=[1, 1, 1,1]):
    in_chennels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_chennels, out_channels])
        biases = tf.get_variable(name='biases', shape=[out_channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, weights, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, biases, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def batch_norm(inputs, is_training=True, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros[inputs.get_shape()[-1]])
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


def max_pool(input, kernel_heigh, kernel_width, stride_heigh, stride_width, name, padding="SAME"):
    return tf.nn.max_pool(input,
                          ksize=[1, kernel_heigh, kernel_width, 1],
                          strides=[1, stride_heigh, stride_width, 1],
                          padding=padding,
                          name=name)


def avg_pool(input, kernel_heigh, kernel_width, stride_heigh, stride_width, name, padding="SAME"):
    return tf.nn.avg_pool(input,
                          ksize=[1, kernel_heigh, kernel_width, 1],
                          strides=[1, stride_heigh, stride_width, 1],
                          padding=padding,
                          name=name)


def relu(input, name):
    return tf.nn.relu(input, name=name)

