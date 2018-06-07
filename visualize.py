import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

from keras.utils.data_utils import Sequence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Convolution2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras import applications
from keras import optimizers
from quiver_engine import server  # https://github.com/keplr-io/quiver

DATA_DIR = "./data/"
NUM_CLASSES = 228
IMAGE_SIZE = 256

conv_base = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
for layer in conv_base.layers[:5]:
    layer.trainable = False
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.load_weights(DATA_DIR + "model.best.256.hdf5")
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
    metrics=["accuracy"]
)

# print(model.summary())
server.launch(
    model,
    input_folder='./data/sample',
    temp_folder='./data/filters'
)

