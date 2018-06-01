import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from keras.utils.data_utils import Sequence
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Convolution2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras import backend as K
from quiver_engine import server  # https://github.com/keplr-io/quiver

DATA_DIR = "./data/"
NUM_CLASSES = 228
IMAGE_SIZE = 256

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

model.load_weights(DATA_DIR + "model.best.256.hdf5")

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print(model.summary())
server.launch(
    model,
    input_folder='./data/sample',
    temp_folder='./data/filters'
)

