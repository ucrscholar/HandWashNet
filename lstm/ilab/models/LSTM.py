from tensorflow_core.python.keras.engine.sequential import Sequential
from tensorflow_core.python.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout
from tensorflow_core.python.keras.layers.normalization import BatchNormalization


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