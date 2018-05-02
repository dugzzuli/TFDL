import tensorflow as tf
import p404_Slim_cnn as model

with tf.Graph().as_default():
    image = tf.random_normal([1, 217, 217, 3])
    probabilities = model.Slim_cnn(image, 5)
    probabilities = tf.nn.softmax(probabilities.net)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(probabilities)

        print("Res Shape:")
        print(res.shape)
        print("\nRes:")
        print(res)
