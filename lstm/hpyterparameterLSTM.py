"""
#This script demonstrates the use of a convolutional LSTM network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""

import os

from tensorflow_core.python import expand_dims

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from keras_preprocessing.image import ImageDataGenerator
import cv2
from numpy import random
import csv

from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Dense, Flatten, AveragePooling3D, Dropout, MaxPooling2D, Lambda
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from ilab.utils import get_augmented
from ilab.utils import MyCustomCallback
from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D, LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape

import numpy as np
import pylab as plt

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

IMG_WIDTH = 50;
IMG_HEIGHT = 50;
IMG_CHANNEL = 3;
CLASS_NUM=52;
BATCH_SIZE = 10;

m_width = 50
m_heigh = 50
batch_size = 10

seq = Sequential()
#seq.add(Lambda(lambda x: expand_dims(x, 0)))
seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                   input_shape=(None, m_width, m_heigh, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                   padding='same', return_sequences=False))
seq.add(BatchNormalization())

'''seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same'))
# seq.add(Flatten())
# seq.add(BatchNormalization())

seq.add(AveragePooling3D(pool_size=(1, m_width, m_heigh), padding='same'))
seq.add(Reshape((-1, batch_size)))
seq.add(Dense(
    units=52,
    activation='sigmoid'))
'''

seq.add(Flatten())
seq.add(Dense(128, activation='relu'))
seq.add(Dropout(0.25))
seq.add(Dense(52, activation='softmax'))
'''
seq.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
seq.add(Flatten())
seq.add(BatchNormalization())
seq.add(Dropout(0.25))

seq.add(Dense(1024, activation='relu'))
seq.add(BatchNormalization())
seq.add(Dropout(0.4))

seq.add(Dense(512, activation='relu'))
seq.add(BatchNormalization())
seq.add(Dropout(0.4))

seq.add(Dense(52, activation='linear'))
'''
seq.compile(loss='categorical_crossentropy', optimizer='adadelta')

seq.summary()
data_path = '/data1/shengjun/db/output/'
dir_list = [os.path.join(data_path, o) for o in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, o))]
train_list = dir_list[0:40]
val_list = dir_list[41:50]
random.shuffle(train_list)
random.shuffle(val_list)




#######################################################
import tensorflow as tf

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
#labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


#######################################################
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
train_sample_num = 0;
verify_sample_num = 0;
for base_dir in val_list:
    rgb_sample_dir = os.path.join(base_dir, 'rgb')
    rgb_filenames = [rgb_file for rgb_file in os.listdir(rgb_sample_dir) if
                     os.path.isfile(os.path.join(rgb_sample_dir, rgb_file))]

    x = rgb_filenames[0][0:-4]
    rgb_filenames.sort(key=lambda annotation: int(annotation[0:-4]))
    verify_sample_num += len(rgb_filenames)


for base_dir in train_list:
    rgb_sample_dir = os.path.join(base_dir, 'rgb')
    rgb_filenames = [rgb_file for rgb_file in os.listdir(rgb_sample_dir) if
                     os.path.isfile(os.path.join(rgb_sample_dir, rgb_file))]

    x = rgb_filenames[0][0:-4]
    rgb_filenames.sort(key=lambda annotation: int(annotation[0:-4]))
    train_sample_num += len(rgb_filenames)

from itertools import groupby
class CSequence(Sequence):
        # return [characters.find(c) for c in text]
    def __init__(self, features, datagen=None, batch_size=10,fname=''):
        # self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.count = 0
        self.sample = []
        self.time = []
        self.xdata = []
        self.ydata = []
        self.videos = features
        self.video_postion = 0;
        self.frame_postion = 0;
        self.datagen = datagen;
        self.video_total = 0;
        self.frame_total = 0;
        self.count = 1000;
        self.steps = int(self.count / self.batch_size)
        your_list=[];
        self.dic1= dict()
        self.dic2 = dict()
        self.dic3 = dict()

        with open(data_path + 'fine_labels.json') as json_file:
            self.data = json.load(json_file)
            with open(data_path + fname, mode='r') as e_file:
                reader = csv.reader(e_file)
                your_list = list(reader)
                for i in range(1,int(len(your_list)/3) ):
                #for i in range(1, 70000):
                    self.sample.append(your_list[i][0])
                    self.time.append(your_list[i][1])
                    self.xdata.append(your_list[i][2])
                    self.ydata.append(your_list[i][3])
            for i in range(0, len(self.xdata)):
                self.dic1[i] = (i % self.steps) * self.batch_size
                self.dic2[i] = (i % self.steps + 1) * self.batch_size
                self.dic3[i] = int(i % self.steps)

            self.inv_map = {v: k for k, v in self.data.items()}
    def text_to_labels(self,text):
        # padding
        # index of 'blank' = label_classes - 1

        return int(self.inv_map[text])

    def __len__(self):
        # return int(np.ceil(len(self.x) / float(self.batch_size)))
        return int(np.ceil(len(self.xdata) / float(self.batch_size)))

    def __getitem__(self, index):
        #if self.dic3[index] ==0:
        'Generate one batch of data'
        # Generate indexes of the batch
        list_sample_batch = self.sample[index * self.batch_size: (index + 1) * self.batch_size]
        list_time_batch = self.time[index * self.batch_size: (index + 1) * self.batch_size]
        list_files_batch = self.xdata[index * self.batch_size: (index + 1) * self.batch_size]
        list_labels_batch = self.ydata[index * self.batch_size: (index + 1) * self.batch_size]
        # list_files_batch = self.xdata[index * self.count: (index + 1) * self.count]
        # list_labels_batch = self.ydata[index * self.count: (index + 1) * self.count]

        # Generate data
        self.X, self.y = self.__data_generation(list_sample_batch,list_time_batch,list_files_batch, list_labels_batch)
        #print('generator yielded a batch %d' % index)
        return self.X,self.y

    def __data_generation(self, list_sample_batch,list_time_batch,list_files_batch, list_labels_batch):
        'Generates data containing batch_size samples'

        count_dups = [sum(1 for _ in group) for _, group in groupby(list_sample_batch)]
        #print(count_dups)

        #unique,counts=np.unique(list_sample_batch,return_counts=True)

        XX = np.zeros((len(count_dups), max(count_dups), IMG_WIDTH, IMG_WIDTH, IMG_CHANNEL), dtype=np.float)
        YY = np.zeros((len(count_dups),CLASS_NUM), dtype=np.float)
        X = []
        y = []
        for sample in range(0, len(count_dups) ):
            for time in range(0,count_dups[sample]):
                #
                img = cv2.imread(list_files_batch[time])
                if img is None:
                    print(list_files_batch[time])
                if self.datagen is not None and time < 0.6 * len(list_files_batch):
                    img = self.datagen.random_transform(img)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = np.reshape(img, (*img.shape, 1))
                img=cv2.resize(img, (m_width, m_heigh))
                XX[int(sample), time, 0:img.shape[0], 0:img.shape[1], 0:img.shape[2]] = img;
                #X.append(img)

                label = list_labels_batch[time]
                YY[sample,:]=to_categorical(self.text_to_labels(label), num_classes=52);
                #y.append(to_categorical(self.text_to_labels(label), num_classes=52))


                '''X = np.array(X, dtype=np.float32)
                X /= 255
                y = np.array(y, dtype=np.float32)
                '''

        #return X[np.newaxis, ::], y
        return XX/255, YY



def generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    for base_dir in features:
        rgb_sample_dir = os.path.join(base_dir, 'rgb')
        rgb_filenames = [rgb_file for rgb_file in os.listdir(rgb_sample_dir) if
                         os.path.isfile(os.path.join(rgb_sample_dir, rgb_file))]
        x = rgb_filenames[0][0:-4]
        rgb_filenames.sort(key=lambda annotation: int(annotation[0:-4]))
        j = 0
        output_coarse_path = os.path.join(base_dir, 'labels_coarse.npy')
        output_fine_path = os.path.join(base_dir, 'labels_fine.npy')
        output_mid_path = os.path.join(base_dir, 'labels_mid.npy')
        output_custom_path = os.path.join(base_dir, 'labels_custom.npy')

        fine_labels = np.load(output_fine_path)
        batch_features = np.zeros((len(rgb_filenames), m_width, m_heigh, 3))
        batch_labels = np.zeros((len(rgb_filenames), 1))

        while j < len(rgb_filenames):
            rgb_file_dir = os.path.join(rgb_sample_dir, rgb_filenames[j])
            im = cv2.imread(rgb_file_dir)
            batch_features[j] = cv2.resize(im, (m_width, m_heigh))
            batch_labels[j] = y_binary = [i for i, x in enumerate(fine_labels[j]) if x]
            j = j + 1
        batch_labels = to_categorical(batch_labels, num_classes=52)
        j = 0
        t1 = batch_features[np.newaxis, ::]
        t2 = batch_labels[np.newaxis, ::]

        while j < len(rgb_filenames):
            if j + batch_size < len(rgb_filenames):
                x = t1[:, j:j + batch_size]
                y = t2[:, j:j + batch_size]
                print('yield:{}:total:{}:name:{}'.format(j,len(rgb_filenames),base_dir))
                yield x, y
            j = j + batch_size

import json
def generatorPanda(features, labels, batch_size,filename):
    # Create empty arrays to contain batch of features and labels#
    with open(data_path + 'fine_labels.json') as json_file:
        data = json.load(json_file)
        with open(data_path + filename, mode='w') as e_file:
            HEADER = ['sample','time','name', 'label']
            csv_writer = csv.writer(e_file)
            csv_writer.writerow(HEADER)
            sample = 0;
            for index, base_dir in enumerate(features):
                rgb_sample_dir = os.path.join(base_dir, 'rgb')
                rgb_filenames = [rgb_file for rgb_file in os.listdir(rgb_sample_dir) if
                                 os.path.isfile(os.path.join(rgb_sample_dir, rgb_file))]
                x = rgb_filenames[0][0:-4]
                rgb_filenames.sort(key=lambda annotation: int(annotation[0:-4]))
                j = 0
                output_coarse_path = os.path.join(base_dir, 'labels_coarse.npy')
                output_fine_path = os.path.join(base_dir, 'labels_fine.npy')
                output_mid_path = os.path.join(base_dir, 'labels_mid.npy')
                output_custom_path = os.path.join(base_dir, 'labels_custom.npy')

                fine_labels = np.load(output_fine_path)
                batch_features = np.zeros((len(rgb_filenames), m_width, m_heigh, 3))
                batch_labels = np.zeros((len(rgb_filenames), 1))

                time = 0;
                oldLabel=-1;
                while j < len(rgb_filenames):
                    rgb_file_dir = os.path.join(rgb_sample_dir, rgb_filenames[j])
                    im = cv2.imread(rgb_file_dir)
                    # batch_features[j] = cv2.resize(im, (m_width, m_heigh))
                    batch_labels[j] = y_binary = [i for i, x in enumerate(fine_labels[j]) if x]
                    if oldLabel == -1:
                        oldLabel = batch_labels[j];
                    elif oldLabel != batch_labels[j]:
                        sample=sample+1;
                        time=0;
                        oldLabel = batch_labels[j];

                    csv_writer.writerow([sample,time, rgb_file_dir, data['{}'.format(int(batch_labels[j][0]))] ])
                    time = time +1
                    print('name:{}'.format(rgb_file_dir))

                    j = j + 1
                j = 0
                # t1 = batch_features[np.newaxis, ::]
                # t2 = batch_labels[np.newaxis, ::]






# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1) ** np.random.randint(0, 2)
                    noisy_movies[i, t,
                    x_shift - w - 1: x_shift + w + 1,
                    y_shift - w - 1: y_shift + w + 1,
                    0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies


# Train the network

from tensorflow.keras.callbacks import ModelCheckpoint

model_filename = 'lstm_model_vsalad4.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

noisy_movies, shifted_movies = generate_movies(n_samples=1200)
'''
train_gen = get_augmented(
    x_train, y_train, batch_size=params['batch_size'],
    data_gen_args=dict(
        rotation_range=5.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    ))
'''
datagens = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.05, # Randomly zoom image
        # width_shift_range=0.08,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        fill_mode="constant",
        cval=250
    )

train_gen = CSequence(train_list,datagen=datagens, batch_size=batch_size,fname='panda.cvs')
val_gen = CSequence(val_list,datagen=datagens, batch_size=batch_size,fname='pandaV.cvs')

#gen = generatorPanda(train_list, 0, batch_size,filename='panda.cvs');
#gen = generatorPanda(val_list, 0, batch_size,filename='pandaV.cvs');

'''
#gen_train = generator(train_list, 0, batch_size);
#gen_val = generator(val_list, 0, batch_size);
import pandas as pd
df=pd.read_csv(data_path+'panda.cvs')
datagen=ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=df,
directory=None,
x_col="name",
y_col="label",
subset="training",
batch_size=10,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(100,100))

valid_generator=datagen.flow_from_dataframe(
dataframe=df,
directory=None,
x_col="name",
y_col="label",
subset="validation",
batch_size=10,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(100,100))
'''
STEP_SIZE_TRAIN=len(train_gen.xdata)//train_gen.batch_size
STEP_SIZE_VALID=len(val_gen.xdata)//val_gen.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history = seq.fit_generator(generator=train_gen,
                            steps_per_epoch=STEP_SIZE_TRAIN,#train_sample_num//batch_size,
                            validation_data=val_gen,
                            validation_steps=STEP_SIZE_VALID,#verify_sample_num//batch_size,
                            shuffle=False,
                            workers=10,
                            use_multiprocessing=True,
                            max_queue_size=50,
                            callbacks=[callback_checkpoint,MyCustomCallback(5)],
                            epochs=50)
# seq.load_weights("C:/Users/sheng/PycharmProjects/unet2/lstm/lstm_model_v1.h5")
# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
# TODO this path has some problem when run in server or local machine
from ilab.utils import plot_segm_history

plot_segm_history(history, metrics=['loss', 'val_loss'], fileName1='loss.png', fileName2='acc.png')

which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])

    for i in range(0, new_pos.shape[1]):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        toplot = new_pos[0, i, ::, ::, 0]
        plt.imshow(toplot)
        ax = fig.add_subplot(122)
        plt.imshow(track[i, ::, ::, 0])
        plt.savefig('{0}_{1}_predict.png'.format((j + 1), (i + 1)))
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
