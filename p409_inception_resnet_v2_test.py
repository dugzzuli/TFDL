'''
http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
'''


import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import global_variable
import inception_resnet_v2 as model

checkpoins_dir = "./data/model/"

with tf.Graph().as_default():
    img = tf.random_normal([1, 299, 299, 3])

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre, _ = model.inception_resnet_v2(img, num_classes=1001, is_training=False)

        model_path = "./data/model/inception_resnet_v2_2016_08_30.ckpt"
        variables = slim.get_model_variables("InceptionResnetV2")
        init_fn = slim.assign_from_checkpoint_fn(model_path, variables)

        with tf.Session() as sess:
            init_fn(sess)
            print((sess.run(pre)))
            print("done")
