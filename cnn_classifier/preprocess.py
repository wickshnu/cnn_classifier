# Python program to create
# Image Classifier using CNN

# Importing the required libraries
import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

"""Setting up the env"""

TRAIN_DIR = "../../data/train"
TEST_DIR = "../../data/test"
IMG_SIZE = 256
LR = 1e-3

"""Labelling the dataset"""


def label_img(img):
    word_label = img.split("_")[0]
    if word_label == "bricks":
        return [1, 0]
    elif word_label == "rebars":
        return [0, 1]


"""Creating the training data"""


def create_train_data():
    # Creating an empty list where we should store the training data
    # after a little preprocessing of the data
    training_data = []

    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):

        # labeling the images
        label = label_img(img)

        path = os.path.join(TRAIN_DIR, img)

        # loading the image from the path and then converting them into
        # grayscale for easier covnet prob
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # final step-forming the training data list with numpy array of the images
        training_data.append([np.array(img), np.array(label)])

    # shuffling of the training data to preserve the random state of our data
    shuffle(training_data)

    # saving our trained data for further uses if required
    np.save("../../data/train_data.npy", training_data)


"""Processing the given test data"""
# Almost same as processing the training data but
# we dont have to label it.
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split(".")[0].split("_")[-1]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save("../../data/test_data.npy", testing_data)


if __name__ == "__main__":
    create_train_data()
#    process_test_data()
