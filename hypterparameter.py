# %%

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
import glob
import os
import sys
from PIL import Image



# %% md

## Load data

# %%

masks = glob.glob("/data1/shengjun/data/handbig/HGR1/*.bmp")
#masks = glob.glob("C:/Users/sheng/Downloads/HGR1/*.bmp")
orgs = list(map(lambda x: x.replace(".bmp", ".jpg"), masks))

# %%

imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((384, 384))))
    masks_list.append(np.array(Image.open(mask).resize((384, 384))))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

# %%

print(imgs_np.shape, masks_np.shape)

# %% md

print(imgs_np.max(), masks_np.max())

# %%

x = np.asarray(imgs_np, dtype=np.float32) / 255
y = np.asarray(masks_np, dtype=np.float32)

# %%

print(x.max(), y.max())

# %%

print(x.shape, y.shape)

# %%

y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
print(x.shape, y.shape)

# %% md

##  Train/val split

# %%

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

leng=len(x_train);

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

# %% md

## Prepare train generator with data augmentation

# %%

from keras_unet.utils import get_augmented



# %% md

import talos
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

from keras_unet.models import custom_unet


# add input parameters to the function
def diabetes(x_train, y_train, x_val, y_val, params):
    # replace the hyperparameter inputs with references to params dictionary

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

    # %%
    input_shape = x_train[0].shape
    sample_batch = next(train_gen)
    xx, yy = sample_batch
    print(xx.shape, yy.shape)

    model = custom_unet(
        input_shape,
        filters=params['filters'],
        use_batch_norm=True,
        dropout=params['dropout'],
        dropout_change_per_layer=0.0,
        num_layers=params['num_layers'],
        output_activation=params['output_activation']
    )

    # %%

    model.summary()

    model.compile(
        optimizer=params['optimizer'],
        # optimizer=SGD(lr=0.01, momentum=0.99),
        loss='binary_crossentropy',
        # loss=jaccard_distance,
        metrics=[iou, iou_thresholded]
    )

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=leng/params['batch_size'],
        epochs=params['epochs'],
        max_queue_size=5,
        validation_data=(x_val, y_val),
        callbacks=[]
    )

    # modify the output model
    return history, model

p = {'activation':['relu', 'elu'],
     'optimizer': ['Adam', 'AdaDelta'],
     'losses': ['logcosh'],
     'shapes': ['brick'],
     'first_neuron': [32],
     'dropout': [.2, .3],
     'batch_size': [1, 2, 3, 4, 5, 6],
     'num_layers':[2,3,4,5],
     'output_activation':['sigmoid' , 'softmax'],
     'filters': [16,32,64,128],
     'epochs': [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]}
## Initialize network
t = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val, params=p, model=diabetes, reduction_metric='val_iou', experiment_name='diabetes')
# %%

