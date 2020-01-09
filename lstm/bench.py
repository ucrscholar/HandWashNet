import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#{}".format(sys.argv[1])
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=False)
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import sys
from itertools import groupby

import imageio
from keras_preprocessing.image import ImageDataGenerator

from ilab.utils import MyCustomCallback


from ilab.models import convLSTM
import pandas as pd
from tensorflow_core.python.keras.utils import Sequence
import numpy as np
import dask
import dask.array as da

IMG_WIDTH = 100;
IMG_HEIGHT = 100;
IMG_CHANNEL = 3;
CLASS_NUM = 52;
BATCH_SIZE = 32;
EPOCH_NUM = 50;



df = pd.read_csv(r"/data1/shengjun/db/output/panda.csv", dtype='str', skiprows= lambda x: (x % 10 != 0 and x!=0))
#df['id'] = df['id'] + '.png'
print(df.head())
print(df.dtypes)

#sample = [sum(1 for _ in group) for _, group in groupby(df['sample'])]
#ms= max(sample)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.25)
train_generator = datagen.flow_from_dataframe(dataframe=df,
                                              #directory=r"C:\\Users\\sheng\\Downloads\\cifar-10\\train\\",
                                              x_col="name", y_col="label", shuffle=False,
                                              subset='training', class_mode="categorical", target_size=(IMG_WIDTH, IMG_HEIGHT),
                                              batch_size=BATCH_SIZE)

validation_generator = datagen.flow_from_dataframe(dataframe=df,
                                                   #directory=r"C:\\Users\\sheng\\Downloads\\cifar-10\\train\\",
                                                   x_col="name", y_col="label", shuffle=False,
                                                   subset='validation', class_mode="categorical", target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                   batch_size=BATCH_SIZE)

# for x,y in validation_generator:
#    print (x)

framesX = []
framesY = []
framesXv = []
framesYv = []
def generator(generator, steps,x,y):
    index = 0;
    while index<steps:
        #if index == 14:
        #    print('hello')
        X, Y = generator[index]
        YY = np.argmax(Y, axis=1)
        count_dups = [sum(1 for _ in group) for _, group in groupby(YY)]
        pos = 0;

        for i, xx in enumerate(count_dups):
            XX = X[pos:pos + xx, ::]
            YYY = Y[pos, ::]
            #framesX.append(XX)
            #framesY.append(YYY)
            x.append(XX[np.newaxis, ::]);
            y.append(YYY[np.newaxis, ::]);
            pos += xx;
        index+=1;

        '''images = np.stack(framesX, axis=0)
        labels = np.stack(framesY, axis=0)
        return images, labels
        yield x_train, y_train'''


STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

train_gen = generator(train_generator,STEP_SIZE_TRAIN,framesX,framesY)
val_gen = generator(validation_generator,STEP_SIZE_VALID,framesXv,framesYv)

from tensorflow.keras.callbacks import ModelCheckpoint

model_filename = 'lstm_model_vsalad12.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

'''history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_gen,
                    validation_steps=STEP_SIZE_VALID,
                    shuffle=False,
                    epochs=EPOCH_NUM,
                    callbacks=[callback_checkpoint,MyCustomCallback(500)])
model.fit(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_gen,
                    validation_steps=STEP_SIZE_VALID,
                    shuffle=False,
                    epochs=EPOCH_NUM,
                    callbacks=[callback_checkpoint,MyCustomCallback(500)])'''

model = convLSTM(IMG_WIDTH, IMG_HEIGHT)
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.summary()

history = model.fit(x=framesX[0:2000], y=framesY[0:2000], epochs=10, verbose=1, callbacks=[callback_checkpoint,MyCustomCallback(500)], validation_split=0.25, validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=3, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)


from ilab.utils import plot_segm_history

plot_segm_history(history, metrics=['loss', 'val_loss'], fileName1='loss13.png', fileName2='acc13.png')