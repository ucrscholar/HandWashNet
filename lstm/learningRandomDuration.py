import os

from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.layers import RepeatVector, BatchNormalization, Dropout
from tensorflow_core.python.keras.utils import to_categorical
import random as rand
from random import randint
from numpy import array
from numpy import zeros
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed

from math import sin, cos, log10, sqrt, ceil, pow
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot

from numpy import zeros
from random import randint
from matplotlib import pyplot

import numpy as np

from ilab.utils import plot_segm_history
from tensorflow.keras.callbacks import ModelCheckpoint
import os.path
from os import path
from tensorflow.keras.utils import multi_gpu_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tag = {0: 'NailWashLeft', 1: 'NailWashRight', 2: 'ThumbFingureWash', 3: 'ForeFingureWash'}
inv_tag = {v: k for k, v in tag.items()}

ROOTPATH = './VRandom2/'
SAMPLESNUM = 4000
ITERATE = 1
# configure problem
SIZE = 50
SHUFF=True
BATCHSIZE = 8

def drawImage(img):
    f = pyplot.figure(figsize=(5, 5))
    # create a grayscale subplot for each frame
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(img, cmap='Greys')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    pyplot.show()
# generate damped sine wave in [0,1]
def generate_DampedSin(period, i, decay, amplify=1):
    return [i / period, 0.5 + 0.5 * amplify * sin(2 * pi * i / period) * exp(-decay * i)]

def generate_DampedSinDuration(period, i, decay, amplify=1):
    if i > period*2:
        return [-1, -1]
    return [i / period, 0.5 + 0.5 * amplify * sin(2 * pi * i / period) * exp(-decay * i)]

def generate_sin(period, i, decay=0, amplify=1):
    return [i / period, 0.5 + 0.5 * amplify * sin(2 * pi * i / period)]

def generate_sinDuration(period, i, decay=0, amplify=1):
    if i>period:
        return [-1,-1]
    return [i / period, 0.5 + 0.5 * amplify * sin(2 * pi * i / period)]


def generate_circle(period, i, decay=0, radius=1, x0=0, y0=0):
    R = radius
    x_0 = x0
    y_0 = y0
    d = uniform(0.01, 0.1)
    # for t in range(0, 2 * pi, 0.01):
    dd = list();
    zz = list();
    for t in range(1000):
        t = t / 1000.0
        x = (R * cos(2 * pi * t / period) + R) / 2 * R + x_0;
        y = (R * sin(2 * pi * t / period) + R) / 2 * R + y_0;
        dd.append(y)
        zz.append(x)
    miy = np.min(dd)
    mix = np.min(zz)
    may = np.max(dd)
    max = np.max(zz)

    x = (R * cos(2 * pi * i / period) + R) / 2 * R + x_0;
    y = (R * sin(2 * pi * i / period) + R) / 2 * R + y_0;

    return [x, y]

def generate_circleDuration(period, i, decay=0, radius=1, x0=0, y0=0):
    R = radius
    x_0 = x0
    y_0 = y0
    d = uniform(0.01, 0.1)
    # for t in range(0, 2 * pi, 0.01):
    dd = list();
    zz = list();
    for t in range(1000):
        t = t / 1000.0
        x = (R * cos(2 * pi * t / period) + R) / 2 * R + x_0;
        y = (R * sin(2 * pi * t / period) + R) / 2 * R + y_0;
        dd.append(y)
        zz.append(x)
    miy = np.min(dd)
    mix = np.min(zz)
    may = np.max(dd)
    max = np.max(zz)

    if i>period:
        return [-1,-1]

    x = (R * cos(2 * pi * i / period) + R) / 2 * R + x_0;
    y = (R * sin(2 * pi * i / period) + R) / 2 * R + y_0;

    return [x, y]

def generate_Heart(period, i, decay):
    x_0 = randint(0, 1)
    y_0 = randint(0, 1)
    d = uniform(0.01, 0.1)
    t = i;
    dd = list();
    zz = list();
    for t in range(1000):
        t = t / 1000.0
        t = 2 * pi * t / period
        x = 16 * pow(sin(t), 3);
        y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t);
        dd.append(y)
        zz.append(x)
    miy = np.min(dd)
    mix = np.min(zz)
    may = np.max(dd)
    max = np.max(zz)

    i = 2 * pi * i / period
    x = 16 * pow(sin(i), 3);
    y = 13 * cos(i) - 5 * cos(2 * i) - 2 * cos(3 * i) - cos(4 * i);

    x = (x - mix) / (max - mix);
    y = (y - miy) / (may - miy);

    return [x, y]


def generate_HeartDuration(period, i, decay):
    x_0 = randint(0, 1)
    y_0 = randint(0, 1)
    d = uniform(0.01, 0.1)
    t = i;
    dd = list();
    zz = list();
    for t in range(1000):
        t = t / 1000.0
        t = 2 * pi * t / period
        x = 16 * pow(sin(t), 3);
        y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t);
        dd.append(y)
        zz.append(x)
    miy = np.min(dd)
    mix = np.min(zz)
    may = np.max(dd)
    max = np.max(zz)

    if i>period:
        return [-1,-1]

    i = 2 * pi * i / period
    x = 16 * pow(sin(i), 3);
    y = 13 * cos(i) - 5 * cos(2 * i) - 2 * cos(3 * i) - cos(4 * i);

    x = (x - mix) / (max - mix);
    y = (y - miy) / (may - miy);

    return [x, y]

# generate input and output pairs of damped sine waves
def generate_examplesX(length, n_patterns, output):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = randint(10, 20)
        d = uniform(0.01, 0.1)
        sequence = [0, 1]  # generate_sequenceDampedSin(length + output, p, d)
        X.append(sequence[:-output])
        y.append(sequence[-output:])
    X = array(X).reshape(n_patterns, length, 1)
    y = array(y).reshape(n_patterns, output)
    return X, y
    # test problem generation


# X, y = generate_examples(20, 5, 5)
# for i in range(len(X)):
#    pyplot.plot([x for x in X[i, :, 0]] + [x for x in y[i]],'-o')
#    pyplot.show()

###########################################################################################
# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
    # define the scope of the next step
    lower = max(0, last_step - 1)
    upper = min(last_frame.shape[0] - 1, last_step + 1)
    # choose the row index for the next step
    step = randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] = 1
    return frame, step


def next_frameSin(row, last_frame, column):
    # define the scope of the next step
    lower = max(0, row - 1)
    upper = min(last_frame.shape[0] - 1, row + 1)
    if row > last_frame.shape[0] - 1:
        row = last_frame.shape[0] - 1
    if column > last_frame.shape[1] - 1:
        column = last_frame.shape[1] - 1
    # choose the row index for the next step
    step = 0#randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    if row>=0 or column>=0:
        frame[row, column] = 1
    return frame, step


def next_frameDampedSin(row, last_frame, column):
    # define the scope of the next step
    lower = max(0, row - 1)
    upper = min(last_frame.shape[0] - 1, row + 1)
    if row > last_frame.shape[0] - 1:
        row = last_frame.shape[0] - 1
    if column > last_frame.shape[1] - 1:
        column = last_frame.shape[1] - 1
    # choose the row index for the next step
    step = 0
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    if row >= 0 or column >= 0:
        frame[row, column] = 1
    return frame, step


def next_frameDampedCircle(row, last_frame, column):
    # define the scope of the next step
    lower = max(0, row - 1)
    upper = min(last_frame.shape[0] - 1, row + 1)
    if row > last_frame.shape[0] - 1:
        row = last_frame.shape[0] - 1
    if column > last_frame.shape[1] - 1:
        column = last_frame.shape[1] - 1
    # choose the row index for the next step
    step = 0  # randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    if row >= 0 or column >= 0:
        frame[row, column] = 1
    return frame, step


def next_frameDampedHeart(row, last_frame, column):
    # define the scope of the next step
    lower = max(0, row - 1)
    upper = min(last_frame.shape[0] - 1, row + 1)
    if row > last_frame.shape[0] - 1:
        row = last_frame.shape[0] - 1
    if column > last_frame.shape[1] - 1:
        column = last_frame.shape[1] - 1
    # choose the row index for the next step
    step = 0  # randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    if row >= 0 or column >= 0:
        frame[row, column] = 1
    return frame, step


# generate a sequence of frames of a dot moving across an image
def build_frames(size, timeStep=0):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()
    # create the first frame
    frame = zeros((size, size))
    step = randint(0, size - 1)
    # decide if we are heading left or right
    right = 1 if rand.random() < 0.5 else 0
    col = 0 if right else size - 1
    frame[step, col] = 0
    frames.append(frame)
    # create all remaining frames
    '''for i in range(1, size):
        col = i if right else size - 1 - i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)'''

    amplify = randint(5, 10) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    labelA.append('NailWashLeft')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_sin(1, i, amplify=amplify)
        # frame = zeros((size, size))
        frame, step = next_frameSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashLeft')
    frame = zeros((size, size))
    frames.append(frame)
    amplify = randint(5, 20) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_DampedSin(0.5, i, 3, amplify=amplify)
        # frame = zeros((size, size))
        frame, step = next_frameDampedSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashRight')

    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    x0 = randint(2, 3) / 10
    y0 = randint(2, 3) / 10
    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_circle(1, i, 0.5, radius=radius, x0=x0, y0=y0)
        # frame = zeros((size, size))
        frame, step = next_frameDampedCircle(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ThumbFingureWash')

    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_Heart(1, i, 0.5)
        # frame = zeros((size, size))
        frame, step = next_frameDampedHeart(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ForeFingureWash')

    return frames, labelA


def GenNailLeft(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    step = randint(0, size - 1)
    # decide if we are heading left or right
    right = 1 if rand.random() < 0.5 else 0
    col = 0 if right else size - 1
    frame[step, col] = 0
    frames.append(frame)

    amplify = randint(5, 10) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    labelA.append('NailWashLeft')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_sin(1, i, amplify=amplify)
        #frame = zeros((size, size))
        frame, step = next_frameSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashLeft')
    return frames, labelA


def GenNailLeftDuration(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    step = randint(0, size - 1)
    # decide if we are heading left or right
    right = 1 if rand.random() < 0.5 else 0
    col = 0 if right else size - 1
    frame[step, col] = 0
    frames.append(frame)

    amplify = randint(5, 10) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    duration = randint(5,10)/10
    labelA.append('NailWashLeft')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_sinDuration(duration, i, amplify=amplify)
        frame = zeros((size, size))
        frame, step = next_frameSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashLeft')
    #drawImage(frame)
    return frames, labelA


def GenNailRight(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    frames.append(frame)
    amplify = randint(5, 20) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_DampedSin(0.5, i, 3, amplify=amplify)
        #frame = zeros((size, size))
        frame, step = next_frameDampedSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashRight')
    return frames, labelA

def GenNailRightDuration(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    frames.append(frame)
    amplify = randint(5, 20) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    duration = randint(3,8)/10

    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_DampedSinDuration(duration, i, 3, amplify=amplify)
        frame = zeros((size, size))
        frame, step = next_frameDampedSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashRight')
    #drawImage(frame)
    return frames, labelA

def GenThumbFinger(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    x0 = randint(2, 3) / 10
    y0 = randint(2, 3) / 10
    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_circle(1, i, 0.5, radius=radius, x0=x0, y0=y0)
        #frame = zeros((size, size))
        frame, step = next_frameDampedCircle(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ThumbFingureWash')
    return frames, labelA


def GenThumbFingerDuration(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    x0 = randint(2, 3) / 10
    y0 = randint(2, 3) / 10
    duration = randint(3,10)/10

    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_circleDuration(duration, i, 0.5, radius=radius, x0=x0, y0=y0)
        frame = zeros((size, size))
        frame, step = next_frameDampedCircle(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ThumbFingureWash')
    #drawImage(frame)
    return frames, labelA

def GenForeFinger(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_Heart(1, i, 0.5)
        #frame = zeros((size, size))
        frame, step = next_frameDampedHeart(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ForeFingureWash')
    return frames, labelA

def GenForeFingerDuration(size):
    frames = list()
    labelA = list()
    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    duration = randint(3,10)/10

    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_HeartDuration(duration, i, 0.5)
        frame = zeros((size, size))
        frame, step = next_frameDampedHeart(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ForeFingureWash')
    #drawImage(frame)
    return frames, labelA


# generate a sequence of frames of a dot moving across an image
def build_frames2(size, timeStep=0):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()
    # create the first frame
    fa, la = GenForeFinger()
    frames += fa
    labelA += la
    fa, la = GenNailLeft(size)
    frames += fa
    labelA += la
    fa, la = GenNailRight()
    frames += fa
    labelA += la
    fa, la = GenThumbFinger()
    frames += fa
    labelA += la

    return frames, labelA


# generate a sequence of frames of a dot moving across an image
def build_framesRandom(size, timeStep=0):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()

    my_list = [GenNailLeft, GenNailRight, GenThumbFinger, GenForeFinger]

    res = [0, 1, 2, 3]
    if SHUFF:
        rand.shuffle(res)
    for i in res:
        fa, la = my_list[res[i]](size)
        frames += fa
        labelA += la

    return frames, labelA


# generate a sequence of frames of a dot moving across an image
def build_framesRandomDuration(size, timeStep=0):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()

    my_list = [GenNailLeftDuration, GenNailRightDuration, GenThumbFingerDuration, GenForeFingerDuration]

    res = [0, 1, 2, 3]
    if SHUFF:
        rand.shuffle(res)
    for i in res:
        fa, la = my_list[res[i]](size)
        frames += fa
        labelA += la

    return frames, labelA





# show the plot
pyplot.show()


def validateData():
    # generate sequence of frames
    size = 30
    frames, right = build_frames(size)
    # plot all feames
    '''
    f=pyplot.figure(figsize=(5,5))
    for seq in range(4):
        for i in range((size ) ):
            # create a grayscale subplot for each frame
            ax=f.add_subplot(1, (size +1) * 4 , (size +1) * seq +i +1)
            ax.imshow(frames[(size ) * seq +i], cmap='Greys')
            # turn of the scale to make it cleaer
            #ax = pyplot.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # show the plot
    pyplot.show()
    pyplot.savefig('fig.png')
    '''

    f, ax = pyplot.subplots(2, (size + 1) * 4, figsize=((size + 1) * 4, 20), sharey=True)
    # make a little extra space between the subplots
    f.subplots_adjust(hspace=0.5)
    # ax[0, 0].set_title("Image A", fontsize=15)
    for i in range((size + 1) * 4):
        ax[1, i].set_axis_off()

    for row in range(0, 1):
        for seq in range(4):
            for i in range((size)):
                ax[row, (size + 1) * seq + i].imshow(frames[(size) * seq + i], cmap='Greys')
                ax[row, (size + 1) * seq + i].set_axis_off()

            ax[row, (size + 1) * seq + i + 1].set_axis_off()
    # pyplot.show()
    # pyplot.savefig('fig.png')


# generate multiple sequences of frames and reshape for network input
def generate_examples(size, n_patterns):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_frames(size)
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = array(X).reshape(n_patterns, size * 4, size, size, 1)
    y = array(y).reshape(n_patterns, 4)
    labels = to_categorical(y, 4)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_examples2(size, n_patterns):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_frames2(size)
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    y = array(y).reshape(n_patterns, 4)
    labels = to_categorical(y, 4)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_examplesRandom(size, n_patterns):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_framesRandom(size)
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    y = array(y).reshape(n_patterns, 4)
    labels = to_categorical(y, 4)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_examplesRandomDuration(size, n_patterns):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_framesRandomDuration(size)
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    y = array(y).reshape(n_patterns, 4)
    labels = to_categorical(y, 4)

    return X, labels


# define the model
'''model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2, 2), activation='relu'), input_shape=(None, size, size, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(size*4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])'''


def models():
    # define LSTM
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (2, 2), activation='relu'), input_shape=(None, SIZE, SIZE, 1)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(75))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(RepeatVector(4))
    model.add(LSTM(50, return_sequences=True))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(4, activation='softmax')))

    # Replicates `model` on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    # parallel_model = multi_gpu_model(model, gpus=[2])
    # parallel_model.compile(loss='categorical_crossentropy',
    #                       optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model;


model = models();

from ilab.models import modelB

model = modelB(SIZE, SIZE);

print(model.summary())
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True, to_file=ROOTPATH + 'modelDemoStandardConvLSTMInception.png')


def fit():
    # fit model
    model_filename = 'lstm_model_vsalad33.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )
    for i in range(100):
        print('begin gen')
        X, y = generate_examples(SIZE, 2000)
        print('begin fit{}/{}'.format(i, 10))

        #if path.exists('lstm_model_vsalad33.h5'):
        #    model.load_weights('lstm_model_vsalad33.h5')
        history = model.fit(X, y, batch_size=BATCHSIZE, epochs=25, validation_split=0.01, shuffle=False,
                            callbacks=[callback_checkpoint])
        plot_segm_history(history, metrics=['loss', 'val_loss'], fileName1='loss33.png', fileName2='acc33.png')


def fit2():
    model_filename = ROOTPATH + 'lstm_model_v1.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )
    # fit model
    for i in range(ITERATE):
        print('begin gen')

        X, y = generate_examplesRandom(SIZE, SAMPLESNUM)
        print('begin fit{}/{}'.format(i, SAMPLESNUM))
        #if path.exists(model_filename):
        #    model.load_weights(model_filename)
        history = model.fit(X, y, batch_size=BATCHSIZE,
                            epochs=1, validation_split=0.01,
                            shuffle=False,
                            workers=2,
                            use_multiprocessing=True,
                            callbacks=[callback_checkpoint])
        plot_segm_history(history, metrics=['loss', 'val_loss'], fileName1=ROOTPATH + 'loss.png',
                          fileName2=ROOTPATH + 'acc.png')


def fitRandomDuration():
    model_filename = ROOTPATH + 'lstm_model_v1.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )
    # fit model
    for i in range(ITERATE):
        print('begin gen')

        X, y = generate_examplesRandomDuration(SIZE, SAMPLESNUM)
        print('begin fit{}/{}'.format(i, SAMPLESNUM))
        #if path.exists(model_filename):
        #    model.load_weights(model_filename)
        history = model.fit(X, y, batch_size=BATCHSIZE,
                            epochs=1, validation_split=0.01,
                            shuffle=False,
                            workers=2,
                            use_multiprocessing=True,
                            callbacks=[callback_checkpoint])
        plot_segm_history(history, metrics=['loss', 'val_loss'], fileName1=ROOTPATH + 'loss.png',
                          fileName2=ROOTPATH + 'acc.png')


# evaluate model
def eval(X, y):
    loss, acc = model.evaluate(X, y, verbose=0,batch_size=BATCHSIZE)
    print('loss: %f, acc: %f' % (loss, acc * 100))


def pred(X, Y):
    # prediction on new data
    yhat = model.predict(X, verbose=0,batch_size=BATCHSIZE)
    expected = [np.argmax(y, axis=1, out=None) for y in Y]
    predicted = predicted = np.argmax(yhat, axis=1)
    print('Expected: %s, Predicted: %s ' % (expected, predicted))


def test():
    model.load_weights('./a1/lstm_model_vsalad33.h5')
    X, y = generate_examples(SIZE, 5)
    eval(X, y)


def test2():
    model.load_weights(ROOTPATH + 'lstm_model_v1.h5')
    X, y = generate_examples2(SIZE, 5)
    eval(X, y)

    for i in range(10):
        X, y = generate_examples2(SIZE, 1)
        pred(X, y)


def testRandom():
    model.load_weights(ROOTPATH + 'lstm_model_v1.h5')
    X, y = generate_examplesRandom(SIZE, 5)
    eval(X, y)

    for i in range(10):
        X, y = generate_examplesRandom(SIZE, 1)
        pred(X, y)


def testRandomDuration():
    model.load_weights(ROOTPATH + 'lstm_model_v1.h5')
    X, y = generate_examplesRandomDuration(SIZE, 5)
    eval(X, y)

    for i in range(10):
        X, y = generate_examplesRandomDuration(SIZE, 1)
        pred(X, y)


if __name__ == "__main__":
    print("Executed when invoked directly")
    # test2();
    # fit2();
    # testRandom();
    fitRandomDuration();
    testRandomDuration();
