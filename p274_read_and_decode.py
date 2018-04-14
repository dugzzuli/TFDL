import tensorflow as tf


def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_exampple = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_exampple,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                           })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)


    image = tf.reshape(image, [227, 227, 3])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      min_after_dequeue=100,
                                                      num_threads=64,
                                                      capacity=200)

    return image_batch, tf.reshape(label_batch, [batch_size])


if __name__ == "__main__":
    image_batch, label_batch= read_and_decode("./data/cat_and_dog/cat_and_dog_train_r.tfrecodes", batch_size=100)


