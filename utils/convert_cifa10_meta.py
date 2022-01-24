import pickle
import glob
import os

import numpy as np
from skimage.io import imsave
from pathlib import Path
import pandas as pd


PIXELS_DIR = "Raw images"

label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck" 
}

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_as_image(img_flat, fname, label):
    """
        Saves a data blob as an image file.
    """

    # consecutive 1024 entries store color channels of 32x32 image
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    folder_name = PIXELS_DIR + "/" + label_dict[label]
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    imsave(os.path.join(folder_name, fname), img)


def main():
    """
        Entry point.
    """

    train_filenames = []
    train_labels = []

    # use "data_batch_*" for just the training set
    for fname in glob.glob("data_batch*"):
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            fname = data[b"filenames"][i].decode()
            label = data[b"labels"][i]

            # save the image and store the label
            save_as_image(img_flat, fname, label)
            train_filenames.append(label_dict[label] + "/" + fname)
            train_labels.append(label)

    train_meta_files = pd.DataFrame({"filename": train_filenames, "class_label": train_labels})
    train_meta_files.to_csv(
        "metadata/train_split.txt",
        header=None,
        sep=" ",
        encoding="utf-8",
        index=False,
    )

    test_filenames = []
    test_labels = []
    # use "data_batch_*" for just the training set
    for fname in glob.glob("test_batch*"):
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            fname = data[b"filenames"][i].decode()
            label = data[b"labels"][i]

            # save the image and store the label
            save_as_image(img_flat, fname, label)
            test_filenames.append(label_dict[label] + "/" + fname)
            test_labels.append(label)

    test_meta_files = pd.DataFrame({"filename": test_filenames, "class_label": test_labels})
    test_meta_files.to_csv(
        "metadata/test_split.txt",
        header=None,
        sep=" ",
        encoding="utf-8",
        index=False,
    )

if __name__ == "__main__":
    main()