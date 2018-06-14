import json
import math
import numpy as np
from PIL import Image
from keras.utils.data_utils import Sequence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG19
from keras.optimizers import Adam
    
NUM_CLASSES = 228
IMAGE_SIZE = 128
DATA_DIR = "../data/"

with open(DATA_DIR + "train.json") as train, open(DATA_DIR + "validation.json") as validation:
    train_json = json.load(train)
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
    def __init__(self, x_set, y_set, batch_size, resize = False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.resize = resize

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = np.empty([len(batch_x), IMAGE_SIZE, IMAGE_SIZE, 3], dtype='uint8')
        for i, path in enumerate(batch_x):
            try:
                if self.resize:
                    img = Image.open(path)
                    img.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
                    image = np.array(img)
                else:
                    image = np.array(Image.open(path))
            except Exception as e:
                print(e)
                output = [1]*(IMAGE_SIZE*IMAGE_SIZE*3)
                output = np.array(output).reshape(IMAGE_SIZE,IMAGE_SIZE,3).astype('uint8')
                image = Image.fromarray(output).convert('RGB')
            images[i, ...] = image
        return images, np.array(batch_y)  

conv_base = VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in conv_base.layers[:5]:
    layer.trainable = False

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = Adam(lr=0.001),
    # optimizer = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False), 
    metrics=["accuracy"]
)

EPOCHS = 15 
BATCH = 26
STEPS = len(train_paths) // BATCH

train_gen = BatchSequence(train_paths, train_labels, BATCH, resize = True)
val_gen = BatchSequence(validation_paths, validation_labels, BATCH, resize = True)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)

lrate = step_decay_schedule(initial_lr=0.001, decay_factor=0.75, step_size=2)

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(
    generator = train_gen,
    validation_data = val_gen,
    epochs = EPOCHS,
    steps_per_epoch = STEPS,
    callbacks = [checkpoint, lrate],
    workers = 3
)