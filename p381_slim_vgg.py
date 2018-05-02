import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg


save_model = "./data/model/vgg16"
if not tf.gfile.Exists(save_model):
    tf.gfile.MakeDirs(save_model)


def get_images_labesl():
    return None, None


images, labels = get_images_labesl()

predictions = vgg.vgg16(images, is_training=True)
slim.losses.softmax_cross_entropy(predictions, labels)
total_loss = slim.losses.get_total_loss()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train_tensor = slim.learning.create_train_op(total_loss, optimizer)

slim.learning.train(train_tensor, save_model)
