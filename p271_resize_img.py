import cv2
import os


def rebuild(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                path = dir + file
                cv2.imwrite(path, resized)
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey(0)


def resize_img(in_dir, out_dir=None, dim=[227, 227]):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        try:
            image = cv2.imread(filepath)
            resized = cv2.resize(image, dim)
            path = out_dir + file
            cv2.imwrite(path, resized)
        except:
            print(filepath)
            os.remove(filepath)
    cv2.waitKey(0)


if __name__ == "__main__":
    #rebuild("./data/cat_and_dog/train/", "./data/cat_and_dog/train_r/")
    resize_img("./data/cat_and_dog/train/", "./data/cat_and_dog/train_r/")






