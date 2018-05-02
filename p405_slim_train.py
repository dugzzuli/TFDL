import tensorflow as tf

import tensorflow.contrib.slim as slim
import p404_Slim_cnn as model
#import global_variable
from datasets import flowers

flowers_data_dir = "./data/flower_photos"
save_model = "./data/model/slim"

def load_batch(dataset, batch_size=32, height=217, width=217, is_training=True):
    import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as provider
    data_provider = provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=1)
    image_raw, label = data_provider.get(['image', 'label'])
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    images_raw, labels = tf.train.batch([image_raw, label], batch_size=batch_size, num_threads=1, capacity=2*batch_size)
    return images_raw, labels

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    dataset = flowers.get_split("train", flowers_data_dir)
    images, labels = load_batch(dataset)

    probabilities = model.Slim_cnn(images, 5)
    probabilities = tf.nn.softmax(probabilities.net)

    one_hot_labels = slim.one_hot_encoding(labels, 5)
    slim.losses.softmax_cross_entropy(probabilities, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    final_loss = slim.learning.train(train_op, logdir=save_model, number_of_steps=100)


