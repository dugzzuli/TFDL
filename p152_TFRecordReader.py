import tensorflow as tf
import cv2

filename = "data/train.tfrecords"
filename_queue = tf.train.string_input_producer([filename])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'image': tf.FixedLenFeature([], tf.string),
                                   })

img = tf.decode_raw(features['image'], tf.uint8)
img = tf.reshape(img, [500, 500, 3])

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
threads = tf.train.start_queue_runners(sess=sess)

img = tf.cast(img, tf.float32) * (1.0 / 128) - 0.5
label = tf.cast(features['label'], tf.int32)

imgcv2 = sess.run(img)
cv2.imshow('cool', imgcv2)
cv2.waitKey()
