import os
import numpy as np
import tensorflow as tf
from skimage import io
import ulibs
import cv2


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images_list, labels_list, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecodes')
    n_samples = len(labels_list)
    writer = tf.python_io.TFRecordWriter(filename)
    print("\nTransform start......(%d in total)", n_samples)
    for i in np.arange(0, n_samples):
        print("HHHA:====>", images_list[i])
        try:
            image = io.imread(images_list[i]) # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels_list[i])
            example = tf.train.Example(features=tf.train.Features(feature={''
                                                                           'label': int64_feature(label),
                                                                           'image_aw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read:", images_list[i])
    writer.close()
    print("Transform done!")

if __name__ == "__main__":
    images_list, labels_list = ulibs.get_file("./data/cat_and_dog/train_r")
    convert_to_tfrecord(images_list, labels_list, "./data/cat_and_dog/", "cat_and_dog_train_r")