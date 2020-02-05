import math
from math import pi

import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

#from tensorflow_core.python.keras.utils import to_categorical
import random

from tensorflow_core.python.keras.utils.np_utils import to_categorical


def generate_DampedSin(period, i, decay, amplify=1):
    return [i / period, 0.5 + 0.5 * amplify * np.sin(2 * pi * i / period) * np.exp(-decay * i)]


def generate_DampedSinDuration(period, i, decay, amplify=1):
    if i > period * 2:
        return [-1, -1]
    return [i / period, 0.5 + 0.5 * amplify * np.sin(2 * pi * i / period) * np.exp(-decay * i)]


def generate_sin(period, i, decay=0, amplify=1):
    return [i / period, 0.5 + 0.5 * amplify * np.sin(2 * pi * i / period)]


def generate_sinDuration(period, i, decay=0, amplify=1):
    if i > period:
        return [-1, -1]
    return [i / period, 0.5 + 0.5 * amplify * np.sin(2 * pi * i / period)]


def generate_circle(period, i, decay=0, radius=1, x0=0, y0=0):
    R = radius
    x_0 = x0
    y_0 = y0
    d = np.uniform(0.01, 0.1)
    # for t in range(0, 2 * pi, 0.01):
    dd = list()
    zz = list()
    for t in range(1000):
        t = t / 1000.0
        x = (R * np.cos(2 * pi * t / period) + R) / 2 * R + x_0
        y = (R * np.sin(2 * pi * t / period) + R) / 2 * R + y_0
        dd.append(y)
        zz.append(x)

    x = (R * np.cos(2 * pi * i / period) + R) / 2 * R + x_0
    y = (R * np.sin(2 * pi * i / period) + R) / 2 * R + y_0

    return [x, y]


def generate_circleDuration(period, i, decay=0, radius=1, x0=0, y0=0):
    R = radius
    x_0 = x0
    y_0 = y0
    d = np.uniform(0.01, 0.1)
    # for t in range(0, 2 * pi, 0.01):
    dd = list()
    zz = list()
    for t in range(1000):
        t = t / 1000.0
        x = (R * np.cos(2 * pi * t / period) + R) / 2 * R + x_0
        y = (R * np.sin(2 * pi * t / period) + R) / 2 * R + y_0
        dd.append(y)
        zz.append(x)
    miy = min(dd)
    mix = min(zz)
    may = max(dd)
    max = max(zz)

    if i > period:
        return [-1, -1]

    x = (R * np.cos(2 * pi * i / period) + R) / 2 * R + x_0
    y = (R * np.sin(2 * pi * i / period) + R) / 2 * R + y_0

    return [x, y]


def generate_Heart(period, i, decay):
    x_0 = np.randint(0, 1)
    y_0 = np.randint(0, 1)
    d = np.uniform(0.01, 0.1)
    t = i
    dd = list()
    zz = list()
    for t in range(1000):
        t = t / 1000.0
        t = 2 * pi * t / period
        x = 16 * pow(np.sin(t), 3)
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        dd.append(y)
        zz.append(x)
    miy = min(dd)
    mix = min(zz)
    may = max(dd)
    maxx = max(zz)

    i = 2 * pi * i / period
    x = 16 * pow(np.sin(i), 3)
    y = 13 * np.cos(i) - 5 * np.cos(2 * i) - 2 * np.cos(3 * i) - np.cos(4 * i)

    x = (x - mix) / (maxx - mix)
    y = (y - miy) / (may - miy)

    return [x, y]


def generate_HeartDuration(period, i, decay):
    x_0 = np.randint(0, 1)
    y_0 = np.randint(0, 1)
    d = np.uniform(0.01, 0.1)
    t = i
    dd = list()
    zz = list()
    for t in range(1000):
        t = t / 1000.0
        t = 2 * pi * t / period
        x = 16 * pow(np.sin(t), 3)
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        dd.append(y)
        zz.append(x)
    miy = min(dd)
    mix = min(zz)
    may = max(dd)
    max = max(zz)

    if i > period:
        return [-1, -1]

    i = 2 * pi * i / period
    x = 16 * pow(np.sin(i), 3)
    y = 13 * np.cos(i) - 5 * np.cos(2 * i) - 2 * np.cos(3 * i) - np.cos(4 * i)

    x = (x - mix) / (max - mix)
    y = (y - miy) / (may - miy)

    return [x, y]


# generate input and output pairs of damped sine waves
def generate_examplesX(length, n_patterns, output):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = np.randint(10, 20)
        d = np.uniform(0.01, 0.1)
        sequence = [0, 1]  # generate_sequenceDampedSin(length + output, p, d)
        X.append(sequence[:-output])
        y.append(sequence[-output:])
    X = np.array(X).reshape(n_patterns, length, 1)
    y = np.array(y).reshape(n_patterns, output)
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
    step = np.randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] = 1
    return frame, step


def generateFrame(row, column, last_frame):
    # define the scope of the next step
    lower = max(0, row - 1)
    upper = min(last_frame.shape[0] - 1, row + 1)
    if row > last_frame.shape[0] - 1:
        row = last_frame.shape[0] - 1
    if column > last_frame.shape[1] - 1:
        column = last_frame.shape[1] - 1
    # choose the row index for the next step
    step = 0  # np.randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    if row >= 0 or column >= 0:
        frame[row, column] = 1
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
    step = 0  # np.randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    if row >= 0 or column >= 0:
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
    step = 0  # np.randint(lower, upper)
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
    step = 0  # np.randint(lower, upper)
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
    frame = np.zeros((size, size))
    step = np.randint(0, size - 1)
    # decide if we are heading left or right
    right = 1 if np.random() < 0.5 else 0
    col = 0 if right else size - 1
    frame[step, col] = 0
    frames.append(frame)
    # create all remaining frames
    '''for i in range(1, size):
        col = i if right else size - 1 - i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)'''

    amplify = np.randint(5, 10) / 10.0
    xratio = np.randint(1, 4)
    yratio = np.randint(1, 4)
    labelA.append('NailWashLeft')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_sin(1, i, amplify=amplify)
        # frame = np.zeros((size, size))
        frame, step = next_frameSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashLeft')
    frame = np.zeros((size, size))
    frames.append(frame)
    amplify = np.randint(5, 20) / 10.0
    xratio = np.randint(1, 4)
    yratio = np.randint(1, 4)
    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_DampedSin(0.5, i, 3, amplify=amplify)
        # frame = np.zeros((size, size))
        frame, step = next_frameDampedSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashRight')

    frame = np.zeros((size, size))
    frames.append(frame)
    radius = np.randint(5, 7) / 10
    xratio = np.randint(1, 3)
    yratio = np.randint(1, 3)
    x0 = np.randint(2, 3) / 10
    y0 = np.randint(2, 3) / 10
    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_circle(1, i, 0.5, radius=radius, x0=x0, y0=y0)
        # frame = np.zeros((size, size))
        frame, step = next_frameDampedCircle(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ThumbFingureWash')

    frame = np.zeros((size, size))
    frames.append(frame)
    radius = np.randint(5, 7) / 10
    xratio = np.randint(1, 3)
    yratio = np.randint(1, 3)
    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_Heart(1, i, 0.5)
        # frame = np.zeros((size, size))
        frame, step = next_frameDampedHeart(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ForeFingureWash')

    return frames, labelA


def GestureA(size, period=100, type=0):
    frames = list()
    labelA = list()

    amplify = np.random.randint(5, 10) / 10.0
    xratio = 2  # rang(1,5)
    yratio = 1  # rang(0.1,1,0.1)
    zratio = size - yratio * size  # rang(0,size - yratio* size)
    if type == 1 or type == 2:
        xratio = np.random.randint(3, 5)  # rang(1,5)
        yratio = np.random.randint(1, 10) / 10.0  # rang(0.1,1,0.1)
        zratio = np.random.randint(0, size - yratio * size)  # rang(0,size - yratio* size)

    labelA.append('GestureA')
    x = list()
    y = list()
    for i in range(0, period):
        x1, y1 = [i, 50 + 50 * np.sin(2 * pi * i / period)]
        x.append(x1)
        y.append(y1)

    x2 = list()
    y2 = list()
    for i in range(0, period, xratio):
        # print(x[i], y[i])
        xx = x[i] / 100 * (size - 1)
        yy = y[i] / 100 * (size - 1) * yratio + zratio

        x2.append(xx)
        y2.append(yy)
        # frame = np.zeros((size, size), dtype=int)
    for i, (xx, yy) in enumerate(zip(x2, y2)):
        # frame = frame.copy()
        if i < len(x2) - 1:
            frame = np.zeros((size, size), dtype=int)
            frame[math.floor(yy), math.floor(xx)] = 1
            frames.append(frame)

        '''f = pyplot.figure(figsize=(5, 5))
        # create a grayscale subplot for each frame
        ax = f.add_subplot(1, 1, 1)
        ax.imshow(frame, cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pyplot.show()'''
    '''if type == 1:
        for i in range(0, size - len(x2)):
            frame = np.zeros((size, size), dtype=int)
            frames.append(frame)'''

    return frames, labelA


def GestureB(size, period=100, type=0):
    frames = list()
    labelA = list()

    amplify = np.random.randint(5, 10) / 10.0
    xratio = 2  # range(2,5)
    yratio = 0.5  # range(0.1,1,0.1)
    zratio = size - yratio * size  # rang(0,size - yratio* size)
    if type == 1 or type == 2:
        xratio = np.random.randint(3, 5)  # rang(1,5)
        yratio = np.random.randint(1, 10) / 10.0  # rang(0.1,1,0.1)
        zratio = np.random.randint(0, size - yratio * size)  # rang(0,size - yratio* size)

    decay = 0.03
    labelA.append('GestureB')
    x = list()
    y = list()
    for i in range(0, period):
        x1, y1 = [i, 50 + 50 * np.sin(2 * pi * i / (period / 2)) * np.exp(-decay * i)]
        x.append(x1)
        y.append(y1)

    x2 = list()
    y2 = list()
    for i in range(0, period, xratio):
        # print(x[i], y[i])
        xx = x[i] / 100 * (size - 1)
        yy = y[i] / 100 * (size - 1) * yratio + zratio
        x2.append(xx)
        y2.append(yy)
        # frame = np.zeros((size, size))
    for xx, yy in zip(x2, y2):
        frame = np.zeros((size, size))
        # frame = frame.copy()

        frame[math.floor(yy), math.floor(xx)] = 1
        frames.append(frame)

        # f = pyplot.figure(figsize=(5, 5))
        # create a grayscale subplot for each frame
        '''ax = f.add_subplot(1, 1, 1)
        ax.imshow(frame, cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pyplot.show()'''

    '''if type == 1:
        for i in range(0, size - len(x2)):
            frame = np.zeros((size, size), dtype=int)
            frames.append(frame)'''

    return frames, labelA


def GestureC(size, period=100, type=0):
    frames = list()
    labelA = list()

    amplify = np.random.randint(5, 10) / 10.0
    xratio = 2
    yratio = 1
    R = 50
    zratio = size - yratio * size  # rang(0,size - yratio* size)
    if type == 1 or type == 2:
        xratio = np.random.randint(3, 5)  # rang(1,5)
        yratio = np.random.randint(1, 10) / 10.0  # rang(0.1,1,0.1)
        zratio = np.random.randint(0, size - yratio * size)  # rang(0,size - yratio* size)
        R = np.random.randint(40, 50)

    labelA.append('GestureC')
    x = list()
    y = list()
    for i in range(0, period):
        x1 = R * np.cos(2 * pi * i / period) + R
        y1 = R * np.sin(2 * pi * i / period) + R
        x.append(x1)
        y.append(y1)

    x2 = list()
    y2 = list()
    for i in range(0, period, xratio):
        # print(x[i], y[i])
        xx = x[i] / 100 * (size - 1)
        yy = y[i] / 100 * (size - 1) * yratio + zratio
        x2.append(xx)
        y2.append(yy)
        # frame = np.zeros((size, size))
    for xx, yy in zip(x2, y2):
        # frame = frame.copy()
        frame = np.zeros((size, size))

        frame[math.floor(yy), math.floor(xx)] = 1
        frames.append(frame)

        # f = pyplot.figure(figsize=(5, 5))
        # create a grayscale subplot for each frame
        '''ax = f.add_subplot(1, 1, 1)
        ax.imshow(frame, cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pyplot.show()'''

    '''if type == 1:
        for i in range(0, size - len(x2)):
            frame = np.zeros((size, size), dtype=int)
            frames.append(frame)'''

    return frames, labelA


def GestureD(size, period=100, type=0):
    frames = list()
    labelA = list()

    amplify = np.random.randint(5, 10) / 10.0
    xratio = 2  # range(1,5)
    yratio = 1  # rang(0.1,1.0.1)
    A = 100
    P = 25
    zratio = size - yratio * size  # rang(0,size - yratio* size)

    if type == 1 or type == 2:
        xratio = np.random.randint(3, 5)  # rang(1,5)
        yratio = np.random.randint(1, 10) / 10.0  # rang(0.1,1,0.1)
        zratio = np.random.randint(0, size - yratio * size)  # rang(0,size - yratio* size)

    labelA.append('GestureD')
    x = list()
    y = list()

    for i in range(0, period):
        x1 = i
        y1 = (A / P) * (P - abs(i % (2 * P) - P))
        x.append(x1)
        y.append(y1)

    x2 = list()
    y2 = list()
    for i in range(0, period, xratio):
        # print(x[i], y[i])
        xx = x[i] / 100 * (size - 1)
        yy = y[i] / 100 * (size - 1) * yratio + zratio
        x2.append(xx)
        y2.append(yy)
        # frame = np.zeros((size, size))
    for xx, yy in zip(x2, y2):
        # frame = frame.copy()
        frame = np.zeros((size, size))

        frame[math.floor(xx), math.floor(yy)] = 1
        frames.append(frame.T)

        # f = pyplot.figure(figsize=(5, 5))
        # create a grayscale subplot for each frame
        '''ax = f.add_subplot(1, 1, 1)
        ax.imshow(frame.T, cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pyplot.show()'''

    '''if type == 1:
        for i in range(0, size - len(x2)):
            frame = np.zeros((size, size), dtype=int)
            frames.append(frame)'''

    return frames, labelA


def GestureBackground(size, period=5, type=0):
    frames = list()
    labelA = list()

    labelA.append('Background')

    for _ in range(0, period):
        frame = np.zeros((size, size))
        frames.append(frame.T)
    return frames, labelA


'''
def GenNailLeftDuration(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    step = np.randint(0, size - 1)
    # decide if we are heading left or right
    right = 1 if np.random() < 0.5 else 0
    col = 0 if right else size - 1
    frame[step, col] = 0
    frames.append(frame)

    amplify = np.randint(5, 10) / 10.0
    xratio = np.randint(1, 4)
    yratio = np.randint(1, 4)
    duration = np.randint(5, 10) / 10
    labelA.append('NailWashLeft')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_sinDuration(duration, i, amplify=amplify)
        frame = np.zeros((size, size))
        frame, step = next_frameSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashLeft')
    # drawImage(frame)
    return frames, labelA


def GenNailRight(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    frames.append(frame)
    amplify = np.randint(5, 20) / 10.0
    xratio = np.randint(1, 4)
    yratio = np.randint(1, 4)
    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_DampedSin(0.5, i, 3, amplify=amplify)
        # frame = np.zeros((size, size))
        frame, step = next_frameDampedSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashRight')
    return frames, labelA


def GenNailRightDuration(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    frames.append(frame)
    amplify = np.randint(5, 20) / 10.0
    xratio = np.randint(1, 4)
    yratio = np.randint(1, 4)
    duration = np.randint(3, 8) / 10

    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column, row = generate_DampedSinDuration(duration, i, 3, amplify=amplify)
        frame = np.zeros((size, size))
        frame, step = next_frameDampedSin(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('NailWashRight')
    # drawImage(frame)
    return frames, labelA


def GenThumbFinger(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    frames.append(frame)
    radius = np.randint(5, 7) / 10
    xratio = np.randint(1, 3)
    yratio = np.randint(1, 3)
    x0 = np.randint(2, 3) / 10
    y0 = np.randint(2, 3) / 10
    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_circle(1, i, 0.5, radius=radius, x0=x0, y0=y0)
        # frame = np.zeros((size, size))
        frame, step = next_frameDampedCircle(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ThumbFingureWash')
    return frames, labelA


def GenThumbFingerDuration(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    frames.append(frame)
    radius = np.randint(5, 7) / 10
    xratio = np.randint(1, 3)
    yratio = np.randint(1, 3)
    x0 = np.randint(2, 3) / 10
    y0 = np.randint(2, 3) / 10
    duration = np.randint(3, 10) / 10

    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_circleDuration(duration, i, 0.5, radius=radius, x0=x0, y0=y0)
        frame = np.zeros((size, size))
        frame, step = next_frameDampedCircle(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ThumbFingureWash')
    # drawImage(frame)
    return frames, labelA


def GenForeFinger(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    frames.append(frame)
    radius = np.randint(5, 7) / 10
    xratio = np.randint(1, 3)
    yratio = np.randint(1, 3)
    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_Heart(1, i, 0.5)
        # frame = np.zeros((size, size))
        frame, step = next_frameDampedHeart(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ForeFingureWash')
    return frames, labelA


def GenForeFingerDuration(size):
    frames = list()
    labelA = list()
    frame = np.zeros((size, size))
    frames.append(frame)
    radius = np.randint(5, 7) / 10
    xratio = np.randint(1, 3)
    yratio = np.randint(1, 3)
    duration = np.randint(3, 10) / 10

    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column, row = generate_HeartDuration(duration, i, 0.5)
        frame = np.zeros((size, size))
        frame, step = next_frameDampedHeart(int(row * size / xratio), frame, int(column * size / yratio))
        frames.append(frame)
        # labelA.append('ForeFingureWash')
    # drawImage(frame)
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
    fa, la = GenNailLeft()
    frames += fa
    labelA += la
    fa, la = GenNailRight()
    frames += fa
    labelA += la
    fa, la = GenThumbFinger()
    frames += fa
    labelA += la

    return frames, labelA
'''


# generate a sequence of frames of a dot moving across an image
def build_frames_DB_A(size, timeStep=0, shuff=False):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()

    my_list = [GestureA, GestureB, GestureC, GestureD]

    res = [0, 1, 2, 3]
    if shuff:
        random.shuffle(res)
    for i in res:
        fa, la = my_list[res[i]](period=80, size=size)
        frames += fa
        labelA += la
        fat = list()
        lat = list()

        if size - len(fa) > 0:
            fa, la = GestureBackground(size, period=size - len(fa))
            frames += fa
            labelA += la

    return frames, labelA


def build_frames_DB_B(size, timeStep=0, shuff=False):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()

    my_list = [GestureA, GestureB, GestureC, GestureD]

    res = [0, 1, 2, 3]
    if shuff:
        random.shuffle(res)
    for i in res:
        fat = list()
        lat = list()
        fa, la = my_list[res[i]](size, type=1)
        frames += fa
        labelA += la
        if size - len(fa) > 0:
            fa, la = GestureBackground(size, period=size - len(fa))
            frames += fa
            labelA += la

    return frames, labelA


def build_frames_DB_C(size, timeStep=0, shuff=False):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()
    fat = list()
    lat = list()
    my_list = [GestureA, GestureB, GestureC, GestureD]

    res = [0, 1, 2, 3]
    if shuff:
        random.shuffle(res)
    for index, i in enumerate(res):
        fa, la = my_list[res[i]](size, type=2)
        frames += fa
        labelA += la
        if index != 3:
            fa, la = GestureBackground(size, period=5)
            frames += fa
            labelA += la

    if size * 4 - len(frames) > 0:
        fa, la = GestureBackground(size, period=size * 4 - len(frames))
        frames += fa
        labelA += la

    return frames, labelA


def build_frames_DB_D(size, timeStep=0, shuff=False):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()

    my_list = [GestureA, GestureB, GestureC, GestureD]

    res = [0, 1, 2, 3]
    if shuff:
        random.shuffle(res)
    for i in res:
        fa, la = my_list[res[i]](size, type=2)
        frames += fa
        labelA += la

    return frames, labelA


# generate a sequence of frames of a dot moving across an image
def build_framesRandomDuration(size, timeStep=0, shuff=False):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()

    my_list = [GenNailLeftDuration, GenNailRightDuration, GenThumbFingerDuration, GenForeFingerDuration]

    res = [0, 1, 2, 3]
    if shuff:
        np.rand.shuffle(res)
    for i in res:
        fa, la = my_list[res[i]](size)
        frames += fa
        labelA += la

    return frames, labelA


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

    X = np.array(X).reshape(n_patterns, size * 4, size, size, 1)
    y = np.array(y).reshape(n_patterns, 4)
    labels = to_categorical(y, 4)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_sample(size, n_patterns, parameter=None):
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

    X = np.array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    y = np.array(y).reshape(n_patterns, 4)
    labels = to_categorical(y, 4)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_DB_A(size, n_patterns, parameter=None):
    X, y = list(), list()
    for i in range(n_patterns):
        print("gen{}/{}".format(i, n_patterns))
        frames, labels = build_frames_DB_A(size=size, shuff=parameter['shuff'][0])
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    #XX = np.array(X)
    #XX.shape = (n_patterns, len(X[0]), size, size, 1)
    X = np.array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    # y = np.array(y).reshape(n_patterns, 8)
    labels = to_categorical(y, 5)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_DB_B(size, n_patterns, parameter=None):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_frames_DB_B(size, shuff=parameter['shuff'][0])
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = np.array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    # y = np.array(y).reshape(n_patterns, 8)
    labels = to_categorical(y, 5)

    return X, labels


# generate multiple sequences of frames and reshape for network input
def generate_DB_C(size, n_patterns, parameter=None):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_frames_DB_C(size, shuff=parameter['shuff'][0])
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = np.array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    # y = np.array(y).reshape(n_patterns, 8)
    labels = to_categorical(y, 5)

    return X, labels


def generate_DB_D(size, n_patterns, parameter=None):
    X, y = list(), list()
    for i in range(n_patterns):
        # print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_frames_DB_D(size, shuff=parameter['shuff'][0])
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = np.array(X).reshape(n_patterns, len(X[0]), size, size, 1)
    # y = np.array(y).reshape(n_patterns, 8)
    labels = to_categorical(y, 5)

    return X, labels
