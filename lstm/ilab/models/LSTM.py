from tensorflow_core.python.keras.engine.sequential import Sequential
from tensorflow_core.python.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout, concatenate
from tensorflow_core.python.keras.layers.normalization import BatchNormalization

from tensorflow_core.python.keras.models import Sequential, Model
from tensorflow_core.python.keras.layers import (ConvLSTM2D, BatchNormalization, Convolution3D, Convolution2D,
                          TimeDistributed, MaxPooling2D, UpSampling2D, Input, merge)

from ilab.utils import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)

def convLSTM(
        m_width,
        m_heigh
            ):
    seq = Sequential()
    # seq.add(Lambda(lambda x: expand_dims(x, 0)))
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
    return seq;




def binary_net(input_shape):
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation='sigmoid',
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss='binary_crossentropy', optimizer='adadelta')
    return net


def class_net(input_shape):
    c = 3
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=4 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=8 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')
    return net


def class_net_ms(input_shape):
    c = 12
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(TimeDistributed(MaxPooling2D((2, 2), (2, 2))))

    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    # net.add(TimeDistributed(MaxPooling2D((2, 2), (2, 2))))

    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    # net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
    #                    border_mode='same', return_sequences=True))
    # net.add(TimeDistributed(UpSampling2D((2, 2))))
    # net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
    #                    border_mode='same', return_sequences=True))

    net.add(TimeDistributed(UpSampling2D((2, 2))))
    net.add(Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')
    return net


def class_net_fcn_1p(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')

    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(input_img)
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    c1 = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    c2 = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = merge([c1, x], mode='concat')
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    output = Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                           kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                           border_mode='same', dim_ordering='tf', name='output')(x)
    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')

    return model


def class_net_fcn_1p_lstm(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = merge([c1, x], mode='concat')

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    # x = TimeDistributed(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 396, 440), border_mode='valid'))(x)

    output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)

    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
    return model


def class_net_fcn_2p_lstm(input_shape):
    c = 32
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = concatenate([c2, x], axis = -1)
    x = TimeDistributed(Convolution2D(c, 3, 3, padding='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #x = concatenate([c1, x], axis = -1)
    # x = TimeDistributed(Convolution2D(c, 3, 3, padding='same'))(x)

    x = TimeDistributed(Convolution2D(3, 3, 3, padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    output = TimeDistributed(Convolution2D(3, 3, 3, padding='same', activation=softmax_2d(-1)))(x)

    model = Model(input_img, output)
    model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
    return model


def class_net_fcn_lstm(input_shape):
    c = 32
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = concatenate([c2, x], axis = -1)
    x = TimeDistributed(Convolution2D(c, 3, 3, padding='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #x = concatenate([c1, x], axis = -1)
    # x = TimeDistributed(Convolution2D(c, 3, 3, padding='same'))(x)

    x = TimeDistributed(Convolution2D(3, 3, 3, padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(filters=32, kernel_size=(1, 1), return_sequences=False)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(52, activation='softmax')(x)
    #output = TimeDistributed(Convolution2D(3, 3, 3, padding='same', activation=softmax_2d(-1)))(x)

    model = Model(input_img, output)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model