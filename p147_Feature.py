import tensorflow as tf
import numpy as np

a_data = 0.834
b_data = [17]

c_data = np.array([[0, 1, 2], [3, 4, 5]])
c = c_data.astype(np.uint8)
c_raw = c.tostring()

writer = tf.python_io.TFRecordWriter("data/dataTest.tfrecords")
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'a': tf.train.Feature(float_list=tf.train.FloatList(value=[a_data])),
            'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b_data)),
            'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=c_data))
        }
    )
)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

writer.write(example.SerializeToString())

writer.close()


