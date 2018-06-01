import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

from keras.utils.data_utils import Sequence
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Convolution2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D

NUM_CLASSES = 228
IMAGE_SIZE = 256

DATA_DIR = "../data_resize/"

with open(DATA_DIR + "train.json") as train, open(DATA_DIR + "test.json") as test, open(DATA_DIR + "validation.json") as validation:
    train_json = json.load(train)
    test_json = json.load(test)
    validation_json = json.load(validation)
    
def generate_paths_and_labels(json_obj, folder):
    paths, labels = [], []
    for data in json_obj['annotations']:
        label = [int(x) for x in data["labelId"]]
        image_path = DATA_DIR + "{}/id_{}_labels_{}.jpg".format(folder, data["imageId"], label)
        paths.append(image_path)
        temp_array = [0] * NUM_CLASSES
        for elem in data['labelId']:
            temp_array[int(elem) - 1] = 1
        labels.append(temp_array)
    return paths, labels

train_paths, train_labels = generate_paths_and_labels(train_json, "train")
validation_paths, validation_labels = generate_paths_and_labels(validation_json, "validation")

class BatchSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = np.empty([len(batch_x), IMAGE_SIZE, IMAGE_SIZE, 3])
        for i, path in enumerate(batch_x):
            try:
                image = np.array(Image.open(path))
            except Exception as e:
                print(e)
                output = [1]*(IMAGE_SIZE*IMAGE_SIZE*3)
                output = np.array(output).reshape(IMAGE_SIZE,IMAGE_SIZE,3).astype('uint8')
                image = Image.fromarray(output).convert('RGB')
            images[i, ...] = image
        return images, np.array(batch_y)    
    
    # def path_to_image(self, path):
    #     try:
    #         image = np.array(Image.open(path))
    #     except Exception as e:
    #         print(e)
    #         output = [1]*(IMAGE_SIZE*IMAGE_SIZE*3)
    #         output = np.array(output).reshape(IMAGE_SIZE,IMAGE_SIZE,3).astype('uint8')
    #         image = Image.fromarray(output).convert('RGB')
    #     return image
        # return resize(np.array(Image.open(path)), (IMAGE_SIZE, IMAGE_SIZE))

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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print(model.summary())

# train_paths = train_paths[:10000]
# train_labels = train_labels[:10000]

EPOCHS = 5
BATCH = 256
STEPS = len(train_paths) // BATCH

train_gen = BatchSequence(train_paths, train_labels, BATCH)
val_gen = BatchSequence(validation_paths, validation_labels, BATCH)

checkpointer = ModelCheckpoint(
    filepath='model.best.hdf5', 
    verbose=1,
    save_best_only=True
)

history = model.fit_generator(
    generator = train_gen,
    validation_data = val_gen,
    epochs = EPOCHS,
    steps_per_epoch = STEPS,
    callbacks = [checkpointer],
)