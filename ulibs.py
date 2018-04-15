import cv2
import os
import numpy as np
import tensorflow as tf


def resize_img(in_dir, out_dir=None, dim=(227, 227)):
    '''参考《TensorFlow深度学习应用实践》p171'''
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        print("HHHA:0====>", filepath)
        try:
            image = cv2.imread(filepath)
            resized = cv2.resize(image, dim)
            path = os.path.join(out_dir, file)
            cv2.imwrite(path, resized)
        except:
            print("【图片无法转换】:", filepath)
            #os.remove(filepath)

    cv2.waitKey(0)


def resize_img_test():
    resize_img("./data/cat_and_dog/train/dog/", "./data/cat_and_dog/train_r/dog/")


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        print("HHHA:0====>root", root)
        print("HHHA:1====>sub_folders", sub_folders)
        #print("HHHA:2====>files", files)
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
        # get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))
        #print(files)
    # assign 10 labes based on the folder names
    labels = []
    for one_folder in temp:
        print("HHHA:3====>", one_folder)
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


def get_file_test():
    image_list, label_list = get_file("./data/cat_and_dog/train_r")


def convert_to_tfrecord(images_list, labels_list, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecodes')
    n_samples = len(labels_list)
    writer = tf.python_io.TFRecordWriter(filename)
    print("\nTransform start......(%d in total)", n_samples)
    for i in np.arange(0, n_samples):
        print("HHHA:====>", images_list[i])
        try:
            image = cv2.imread(images_list[i]) # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels_list[i])
            example = tf.train.Example(features=tf.train.Features(
                                                feature={''
                                                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                                                    'image_aw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                                                }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read:", images_list[i])
    writer.close()
    print("Transform done!")


def convert_to_tfrecord_test():
    images_list, labels_list = get_file("./data/cat_and_dog/train_r")
    convert_to_tfrecord(images_list, labels_list, "./data/cat_and_dog/", "cat_and_dog_train_r")



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


def read_and_decode_test():
    image_batch, label_batch= read_and_decode("./data/cat_and_dog/cat_and_dog_train_r.tfrecodes", batch_size=100)


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


def get_batch_test():
    images_list, labels_list = get_file("./data/cat_and_dog/train_r")
    image_batch, label_batch = get_batch(images_list, labels_list, 227, 227, 50, 200)

def onehot(labels):
    """one-hot 编码"""
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


if __name__ == "__main__":
    #convert_to_tfrecord_test()
    read_and_decode_test()

