import cv2
import os
import numpy as np
import tensorflow as tf
import io


def get_file(file_dir):
    """获取图片数据文件位置和图片标签"""
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

if __name__ == "__main__":
    image_list, label_list = get_file("./data/cat_and_dog/train_r")
    #print(image_list)
    #print(label_list)
