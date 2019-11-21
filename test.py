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

masks = glob.glob("C:/videos/ilab_Gesture G_2019-11-20_16-07-44_top/ilab_Gesture G_2019-11-20_16-07-44_top_Depth_*.png")
orgs = list(map(lambda x: x.replace("Depth", "Color"), masks))
#masks = glob.glob("C:/Users/sheng/Downloads/HGR1/*.bmp")
#orgs = list(map(lambda x: x.replace("bmp", "jpg"), masks))

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

## Plot images + masks + overlay (mask over original)

# %%

from keras_unet.utils import plot_imgs

#plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=60)

# %% md

## Get data into correct shape, dtype and range (0.0-1.0)

# %%

print(imgs_np.max(), masks_np.max())

# %%

x = np.asarray(imgs_np, dtype=np.float32) / 255
y = np.asarray(masks_np, dtype=np.float32)

# %%

print(x.max(), y.max())

# %%

print(x.shape, y.shape)

# %%

#y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
print(x.shape, y.shape)

# %% md

##  Train/val split

# %%

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

# %% md

## Prepare train generator with data augmentation

# %%


# %% md

## Initialize network

# %%

from keras_unet.models import custom_unet

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    filters=64,
    use_batch_norm=True,
    dropout=0.2,
    dropout_change_per_layer=0.0,
    num_layers=5,
    output_activation='sigmoid'
)

# %%

model.summary()

# %% md

## Compile + train

# %%

from keras.callbacks import ModelCheckpoint

model_filename = './HDR3/segm_model_v4.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

# %%

from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

model.compile(
    optimizer='AdaDelta',
    # optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    # loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)


model.load_weights(model_filename)
y_pred = model.predict(x_val)

# %%

from keras_unet.utils import plot_imgs

plot_imgs(fileName='ilab_Gesture G_2019-11-20_16-07-44_top.png',org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=30)

# %%
