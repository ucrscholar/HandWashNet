import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import json, codecs


# Runtime data augmentation
def get_augmented(
        X_train,
        Y_train,
        X_val=None,
        Y_val=None,
        batch_size=32,
        seed=0,
        data_gen_args=dict(
            rotation_range=10.,
            # width_shift_range=0.02,
            height_shift_range=0.02,
            shear_range=5,
            # zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='constant'
        )):
    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    train_generator = zip(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=True, seed=seed)
        Y_datagen_val.fit(Y_val, augment=True, seed=seed)
        X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
        Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

        # combine generators into one which yields image and masks
        val_generator = zip(X_val_augmented, Y_val_augmented)

        return train_generator, val_generator
    else:
        return train_generator


def plot_segm_history(history, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss'], fileName1='temp1.png',
                      fileName2='temp2.png'):
    # summarize history for iou
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.legend(metrics, loc='center right', fontsize=15)
    plt.savefig(fileName1)
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(12, 6))
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    # plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.legend(losses, loc='center right', fontsize=15)
    plt.savefig(fileName2)
    # plt.show()


def mask_to_red(mask):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size, img_size)
    c2 = np.zeros((img_size, img_size))
    c3 = np.zeros((img_size, img_size))
    c4 = mask.reshape(img_size, img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def depth_to_blue(depth):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    r, g, b = depth[:, :, 0], depth[:, :, 1], depth[:, :, 2]  # For RGB image
    b[b < 0.8] = 0
    img_size = r.shape[0]
    c1 = np.zeros((img_size, img_size))
    c2 = np.zeros((img_size, img_size))
    c3 = b.reshape(img_size, img_size)
    c4 = b.reshape(img_size, img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def mask_to_rgba(mask, color='red'):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    zeros = np.zeros((img_size, img_size))
    ones = mask.reshape(img_size, img_size)
    if color == 'red':
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == 'green':
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == 'blue':
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == 'yellow':
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == 'magenta':
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == 'cyan':
        return np.stack((zeros, ones, ones, ones), axis=-1)


def plot_imgs(org_imgs,
              mask_imgs,
              fileName='temp.png',
              pred_imgs=None,
              nm_img_to_plot=10,
              figsize=4,
              alpha=0.5
              ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)),
                              cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)),
                              cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1
    plt.savefig(fileName)
    # plt.show()


def featureMaps(feature_maps, square=8, figsize=6):
    im_id = 0
    fig, axes = plt.subplots(square, square, figsize=(square * figsize, square * figsize))
    for m in range(square):
        axes[0, m].set_title(str(m), fontsize=15)

    for m in range(square):
        for n in range(square):
            axes[m, n].imshow(feature_maps[0, :, :, im_id], cmap='gray')
            axes[m, n].set_axis_off()
            im_id += 1
    # plt.savefig(fileName)
    plt.show()


def filter_maps(filters, n_filters=8, m_channel=3, figsize=6):
    im_id = 0
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    fig, axes = plt.subplots(n_filters, m_channel, figsize=(n_filters * figsize, m_channel * figsize))
    for m in range(m_channel):
        axes[0, m].set_title(str(m), fontsize=15)

    for m in range(n_filters):
        for n in range(m_channel):
            axes[m, n].imshow(filters[:, :, n, m], cmap='gray')
            axes[m, n].set_axis_off()
            im_id += 1
    # plt.savefig(fileName)
    plt.show()


def plot_ResultImgs(org_imgs,
                    mask_imgs,
                    fileName='temp.png',
                    pred_imgs=None,
                    nm_img_to_plot=10,
                    figsize=4,
                    alpha=0.5
                    ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 5
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("Depth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("Depth segmentation", fontsize=15)
        axes[0, 4].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()

            axes[m, 3].imshow(depth_to_blue(mask_imgs[im_id]), cmap=get_cmap(mask_imgs))
            axes[m, 3].set_axis_off()

            axes[m, 4].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 4].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)),
                              cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 4].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)),
                              cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1
    plt.savefig(fileName)
    # plt.show()


def zero_pad_mask(mask, desired_size):
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'


def get_patches(img_arr, size=256, stride=256):
    '''
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    '''
    # check size and stride
    if size % stride != 0:
        raise ValueError('size % stride must be equal 0')

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[i * stride:i * stride + size,
                    j * stride:j * stride + size
                    ])

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[i * stride:i * stride + size,
                        j * stride:j * stride + size
                        ])

    else:
        raise ValueError('img_arr.ndim must be equal 3 or 4')

    return np.stack(patches_list)


def plot_patches(img_arr, org_img_size, stride=None, size=None):
    '''
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image
    '''

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError('org_image_size must be a tuple')

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1


########################

def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError('org_image_size must be a tuple')

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = (img_arr.shape[0] // (i_max ** 2))
    nm_images = img_arr.shape[0]

    averaging_value = (size // stride)
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros((org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype)

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[i * stride:i * stride + size, j * stride:j * stride + size, layer] = img_arr[kk, :, :, layer]

                kk += 1
                # TODO add averaging for masks - right now it's just overwritting

        #         for layer in range(nm_layers):
        #             # average some more because overlapping 4 patches
        #             img_bg[stride:i_max*stride, stride:i_max*stride, layer] //= averaging_value
        #             # corners:
        #             img_bg[0:stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value
        #             img_bg[0:stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value

        images_list.append(img_bg)

    return np.stack(images_list)


def saveHist(path, history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] = history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as file:
        json.dump(new_hist, file, separators=(',', ':'), sort_keys=True, indent=4)


def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as file:
        n = json.loads(file.read())
    return n

import smtplib
def notify(name,content):
    fromaddr = 'shezhang@ucr.edu'
    toaddrs = 'shengjun.zhang@ucr.edu'
    msg = "\r\n".join([
        "From: shezhang@ucr.edu",
        "To: shengjun.zhang@ucr.edu",
        "Subject: Progress Report-{}".format(name),
        "",
        "Why, oh why\r\n{}".format(content)
    ])
    username = 'shezhang@ucr.edu'
    password = 'Broadcreate.com139822'
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()

import datetime
class MyCustomCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, total=0):
        self.total= total;

    def on_epoch_end(self, epoch, logs=None):
        x='Training: epoch {}/{} begins at {}'.format(epoch, self.total, datetime.datetime.now().time())
        notify('convLSTM', x)

class Notify():
    def __init__(self, total=0):
        self.total= total;
        notify('empty GPU', 0)



import numpy as np
from tensorflow.keras import backend as K


def to_categorical(y, nb_classes=None):
    """ Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    :param y: label image with values between 0 and 255 (ex. shape: (h, w))
    :param nb_classes: number of classes
    :return: "binary" label image with as many channels as number of labels (ex. shape: (h, w, #lab))
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1

    yy = y.flatten()
    Y = np.zeros((len(yy), nb_classes))
    for i in range(len(yy)):
        Y[i, yy[i]] = 1.
    Y = np.reshape(Y, list(y.shape) + [nb_classes])
    return Y


def softmax_3d(class_dim=-1):
    """ 3D extension of softmax, class is last dim"""
    def activation(x):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=class_dim, keepdims=True))
            s = K.sum(e, axis=class_dim, keepdims=True)
            return e / s
        else:
            raise Exception('Cannot apply softmax to a tensor that is not 2D or '
                            '3D. Here, ndim=' + str(ndim))
    return activation


def categorical_crossentropy_3d_w(alpha, class_dim=-1):
    """ Weighted 3D extension CCE, class is last dim"""
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = y_true * K.log(y_pred)
        cce = K.sum(cce, axis=class_dim)
        cce *= 1 + alpha * K.clip(K.cast(K.argmax(y_true, axis=class_dim), K.floatx()), 0, 1)
        cce = -K.sum(K.sum(cce, axis=-1), axis=-1)
        return cce
    return loss


def categorical_crossentropy_3d(class_dim=-1):
    """2D categorical crossentropy loss
    """
    def loss(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        # avoid numerical instability with _EPSILON clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = -K.sum(K.sum(K.sum(y_true * K.log(y_pred), axis=class_dim), axis=-1), axis=-1)
        return cce
    return loss


def softmax_2d(class_dim=-1):
    """2D softmax activation
    """
    def activation(x):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=class_dim, keepdims=True))
            s = K.sum(e, axis=class_dim, keepdims=True)
            return e / s
        else:
            raise Exception('Cannot apply softmax to a tensor that is not 2D or '
                            '3D. Here, ndim=' + str(ndim))
    return activation


def categorical_crossentropy_2d(class_dim=-1):
    """2D categorical crossentropy loss
    """
    def loss(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        # avoid numerical instability with _EPSILON clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = -K.sum(K.sum(y_true * K.log(y_pred), axis=class_dim), axis=-1)
        return cce
    return loss


def categorical_crossentropy_2d_w(alpha, class_dim=-1):
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = y_true * K.log(y_pred)
        cce = K.sum(cce, axis=class_dim)
        cce *= 1 + alpha * K.clip(K.argmax(y_true, axis=class_dim), 0, 1)
        cce = -K.sum(cce, axis=-1)
        return cce
    return loss
