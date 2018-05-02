import tensorflow.contrib.slim as slim


def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = slim.relu,
                        weights_initializer = slim.xavier_initializer(),
                        weights_reqularizer = slim.l2_regularizer(0.0005)):

        out = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        out = slim.max_pool2d(out, [2, 2], scope='pool1')
        out = slim.repeat(out, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        out = slim.max_pool2d(out, [2, 2], scope='pool2]')
        out = slim.repeat(out, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        out = slim.max_pool2d(out, [2, 2], scope='pool3')
        out = slim.repeat(out, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        out = slim.max_pool2d(out, [2, 2], scope='pool4')
        out = slim.repeat(out, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        out = slim.max_pool2d(out, [2, 2], scope='pool5')
        out = slim.fully_connected(out, 4096, scope='fc6')
        out = slim.dropout(out, scope='droupout6')
        out = slim.fully_connected(out, scope='fc7')
        out = slim.dropout(out, scope='droupout7')
        out = slim.fully_connected(out, 1000, activation_fn=None, scope='fc8')
        res = slim.softmax(out)
    return res


