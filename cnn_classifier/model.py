# Importing the required libraries
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression

IMG_SIZE = 256
LR = 5e-4

"""Setting up the model which will help with tensorflow models"""
model_type = "6conv-basic"
MODEL_NAME = f"bricksvsrebars-{LR}-{model_type}.model"

train_data = np.load("../../data/train_data.npy", allow_pickle=True)
#test = np.load("../../data/test_data.npy", allow_pickle=True)

train = train_data[:int(len(train_data)*0.8)]
test = train_data[int(len(train_data)*0.8):]


"""Setting up the features and labels"""
# X-Features & Y-Labels

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])


"""Creating the neural network using tensorflow"""
tf.compat.v1.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name="input")

convnet = conv_2d(convnet, 32, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation="relu")
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation="softmax")
convnet = regression(
    convnet,
    optimizer="adam",
    learning_rate=LR,
    loss="categorical_crossentropy",
    name="targets",
)

model = tflearn.DNN(convnet, tensorboard_dir="log")

"""Fitting the data into our model"""
# epoch = 5 taken
model.fit(
    {"input": X},
    {"targets": Y},
    n_epoch=30,
    validation_set=({"input": test_x}, {"targets": test_y}),
    snapshot_step=500,
    show_metric=True,
    run_id=MODEL_NAME,
)
model.save(MODEL_NAME)
