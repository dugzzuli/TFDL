import tensorflow as tf
from datasets import dataset_utils

#url = "http://download.tensorflow.org/data/flowers.tar.gz"
url = "http://download.tensorflow.org/data/flowers.tar.gz"
flowers_data_dir = "./data/flower_photos"

if not tf.gfile.Exists(flowers_data_dir):
    tf.gfile.MakeDirs(flowers_data_dir)

dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)
