import os
import tensorflow as tf
from PIL import Image

path = '.\data\jpg'
filename = os.listdir(path)
writer = tf.python_io.TFRecordWriter("train.tfrecords")

for name in os.listdir(path):
    class_path = path + os.sep + name
    for img_name in os.listdir(class_path):

        img_path = class_path + os.sep + img_name
        print(img_path)
        img = Image.open(img_path)
        img = img.resize((500, 500))
        img_raw = img.tobytes()
        #img_raw = Image.open(class_path + os.sep + img_name).resize((500, 500)).tobytes()
        lable = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)]))
        image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        feature = {
            "lable": lable,
            "image": image
            }
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        # example = tf.train.Example(features=tf.train.Features(feature={
        #     "lable": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
        #     "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        # }))
        writer.write(example.SerializeToString())

