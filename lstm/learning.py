import os

from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.layers import RepeatVector, BatchNormalization, Dropout
from tensorflow_core.python.keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



from random import random
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

from math import sin, cos, log10, sqrt, ceil,pow
from math import pi
from math import exp
from random import random
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot

from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot

import  numpy as np

tag= {0:'NailWashLeft',1:'NailWashRight',2:'ThumbFingureWash',3:'ForeFingureWash'}
inv_tag = {v: k for k, v in tag.items()}
# generate damped sine wave in [0,1]
def generate_DampedSin(period, i, decay,amplify=1):
    return [i/ period, 0.5 + 0.5 * amplify*sin(2 * pi * i / period) * exp(-decay * i)]


def generate_sin(period, i, decay=0,amplify=1):
    return [i/period, 0.5+0.5* amplify*sin(2 * pi * i / period)]


def generate_circle(period,i, decay=0,radius=1,x0=0,y0=0):
    R = radius
    x_0 = x0
    y_0 = y0
    d = uniform(0.01, 0.1)
    #for t in range(0, 2 * pi, 0.01):
    dd = list();
    zz = list();
    for t in range(1000):
        t=t/1000.0
        x = (R * cos(2 * pi * t / period) +R)/2*R + x_0;
        y = (R * sin(2 * pi * t / period) +R)/2*R+ y_0;
        dd.append(y)
        zz.append(x)
    miy = np.min(dd)
    mix = np.min(zz)
    may = np.max(dd)
    max = np.max(zz)

    x = (R * cos(2 * pi * i / period) +R) / 2 * R+ x_0;
    y = (R * sin(2 * pi * i / period) +R) / 2 * R+ y_0;


    return [x,y]

def generate_Heart(period, i, decay):
    x_0 = randint(0, 1)
    y_0 = randint(0, 1)
    d = uniform(0.01, 0.1)
    t = i;
    dd = list();
    zz= list();
    for t in range(1000):
        t = t / 1000.0
        t=2 * pi * t / period
        x = 16*pow(sin(t),3);
        y = 13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t);
        dd.append(y)
        zz.append(x)
    miy=np.min(dd)
    mix=np.min(zz)
    may=np.max(dd)
    max=np.max(zz)

    i = 2 * pi * i / period
    x = 16 * pow(sin(i), 3);
    y = 13 * cos(i) - 5 * cos(2 * i) - 2 * cos(3 * i) - cos(4 * i);

    x=(x-mix)/(max-mix);
    y=(y-miy)/(may-miy);


    return [x,y]


# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
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
    step = randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
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
    step = 0#randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
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
    step = 0#randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[row, column] = 1
    return frame, step


# generate a sequence of frames of a dot moving across an image
def build_frames(size,timeStep=0):
    frames = list()
    labelA = list()
    labelB = list()
    labelC = list()
    # create the first frame
    frame = zeros((size, size))
    step = randint(0, size - 1)
    # decide if we are heading left or right
    right = 1 if random() < 0.5 else 0
    col = 0 if right else size - 1
    frame[step, col] = 0
    frames.append(frame)
    # create all remaining frames
    '''for i in range(1, size):
        col = i if right else size - 1 - i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)'''

    amplify = randint(5, 10)/10.0
    xratio =randint(1, 4)
    yratio = randint(1, 4)
    labelA.append('NailWashLeft')
    for i in range(1, size):
        i=i/float(size)
        column,row = generate_sin(1, i, amplify=amplify)
        frame = zeros((size, size))
        frame, step = next_frameSin(int(row*size/xratio), frame, int(column*size/yratio))
        frames.append(frame)
        #labelA.append('NailWashLeft')
    frame = zeros((size, size))
    frames.append(frame)
    amplify = randint(5, 20) / 10.0
    xratio = randint(1, 4)
    yratio = randint(1, 4)
    labelA.append('NailWashRight')
    for i in range(1, size):
        i = i / float(size)
        column,row = generate_DampedSin(0.5, i, 3,amplify=amplify)
        frame = zeros((size, size))
        frame, step = next_frameDampedSin(int(row*size/xratio), frame, int(column*size/yratio))
        frames.append(frame)
        #labelA.append('NailWashRight')

    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7)/10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    x0 = randint(2, 3)/10
    y0 = randint(2, 3)/10
    labelA.append('ThumbFingureWash')
    for i in range(1, size):
        i=float(i)/float(size)
        column,row = generate_circle(1, i, 0.5,radius = radius,x0=x0,y0=y0)
        frame = zeros((size, size))
        frame, step = next_frameDampedCircle(int(row*size/xratio), frame, int(column*size/yratio))
        frames.append(frame)
        #labelA.append('ThumbFingureWash')

    frame = zeros((size, size))
    frames.append(frame)
    radius = randint(5, 7) / 10
    xratio = randint(1, 3)
    yratio = randint(1, 3)
    labelA.append('ForeFingureWash')
    for i in range(1, size):
        i = float(i) / float(size)
        column,row = generate_Heart(1, i, 0.5)
        frame = zeros((size, size))
        frame, step = next_frameDampedHeart(int(row*size/xratio), frame, int(column*size/yratio))
        frames.append(frame)
        #labelA.append('ForeFingureWash')

    return frames, labelA


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

f, ax = pyplot.subplots(2, (size +1) * 4 ,figsize=((size +1) * 4 , 20), sharey=True)
# make a little extra space between the subplots
f.subplots_adjust(hspace=0.5)
#ax[0, 0].set_title("Image A", fontsize=15)
for i in range((size +1) * 4):
    ax[1, i].set_axis_off()

for row in range(0, 1):
    for seq in range(4):
        for i in range((size)):
            ax[row, (size +1) * seq +i ].imshow(frames[(size) * seq + i], cmap='Greys')
            ax[row, (size +1) * seq +i ].set_axis_off()

        ax[row, (size+1) * seq +i+1].set_axis_off()
#pyplot.show()
#pyplot.savefig('fig.png')

# generate multiple sequences of frames and reshape for network input
def generate_examples(size, n_patterns):
    X, y = list(), list()
    for i in range(n_patterns):
        #print("gen{}/{}".format(i,n_patterns))
        frames, labels = build_frames(size)
        code = np.array(labels)
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(code)

        X.append(frames)
        y.append(vec)
    # resize as [samples, timesteps, width, height, channels]

    X = array(X).reshape(n_patterns, size*4, size, size, 1)
    y = array(y).reshape(n_patterns, 4)
    labels=to_categorical(y,4)

    return X, labels


# configure problem
size = 50
# define the model
'''model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2, 2), activation='relu'), input_shape=(None, size, size, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(size*4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])'''


# define LSTM
model = Sequential()
model.add(TimeDistributed(Conv2D(16, (2, 2), activation='relu'), input_shape=(None, size, size, 1)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(Dropout(0.25))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(75))
#model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(RepeatVector(4))
model.add(LSTM(50, return_sequences=True))
#model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(4, activation= 'softmax' )))
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
print(model.summary())
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes = True, to_file='modelWH.png')


from ilab.utils import plot_segm_history
from tensorflow.keras.callbacks import ModelCheckpoint
import os.path
from os import path

if path.exists('lstm_model_vsalad33.h5'):
    os.remove("lstm_model_vsalad33.h5")

model_filename = 'lstm_model_vsalad33.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

# fit model
for i in range(100):
    print('begin gen')
    X, y = generate_examples(size, 2000)
    print('begin fit{}/{}'.format(i,10))
    if path.exists('lstm_model_vsalad33.h5'):
        model.load_weights('lstm_model_vsalad33.h5')
    history = model.fit(X, y, batch_size=32, epochs=25,validation_split=0.25,shuffle=False,callbacks=[callback_checkpoint])
    plot_segm_history(history, metrics=['loss', 'val_loss'], fileName1='loss33.png', fileName2='acc33.png')
# evaluate model
X, y = generate_examples(size, 500)
loss, acc = model.evaluate(X, y, verbose=0)
print('loss: %f, acc: %f' % (loss, acc * 100))
for i in range(10):
    # prediction on new data
    X, Y = generate_examples(size, 1)
    yhat = model.predict_classes(X, verbose=0)
    expected = [np.argmax(y, axis=1, out=None) for y in Y]
    predicted = yhat
    print('Expected: %s, Predicted: %s ' % (expected, predicted))

