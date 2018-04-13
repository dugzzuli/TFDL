import cv2
import os
import numpy as np
import tensorflow as tf
import io

def rebuild(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                path = ".data/cat_and_dog//dog_r/" + file
                cv2.imwrite(path, resized)
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey(0)


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
        # get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))
        print(files)
    # assign 10 labes based on the folder names
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1]
        if letter == 'cat':
            labels = np.append(labels, n_img*[0])
        else:
            labels = np.append(labels, n_img*[1])

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images_list, labels_list, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecodes')
    n_samples = len(labels_list)
    writer = tf.python_io.TFRecodWriter(filename)
    print("\nTransform start......")
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images_list[i]) # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels_list[i])
            example = tf.train.Example(features=tf.train.Features(feature={''
                                                                           'label':int64_feature(label),
                                                                           'image_aw':bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read:", images_list[i])
    writer.close()
    print("Transform done!")


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

    image = tf.reshap(image[227, 227, 3])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      min_after_dequeue=100,
                                                      num_threads=64,
                                                      capacity = 200)

    return image_batch, tf.reshape(label_batch, [batch_size])


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])

    image = tf.image.resize_image_with_crop_or_pad()
    image= tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacit=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


def onehot(labels):
    '''one-hot 编码'''
    n_sample= len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels




