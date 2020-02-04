from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.engine.sequential import Sequential
from tensorflow_core.python.keras.layers import TimeDistributed, Dropout, BatchNormalization, Conv2D, LSTM, \
    RepeatVector, Dense, concatenate, add, Embedding, ConvLSTM2D
from tensorflow_core.python.keras.layers.core import Flatten, Activation
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D


def modelStandard(input_shape, parameter=None):
    # define LSTM
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (2, 2), activation='relu'), input_shape=input_shape))
    model.add(Dropout(parameter['dropout']))
    model.add(BatchNormalization())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    model.add(Dropout(parameter['dropout']))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(parameter['cell1']))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(RepeatVector(8))
    model.add(LSTM(parameter['cell2'], return_sequences=True))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(5, activation='softmax')))

    # Replicates `model` on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    # parallel_model = multi_gpu_model(model, gpus=[2])
    # parallel_model.compile(loss='categorical_crossentropy',
    #                       optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model;


def modelStandardB(row, col):
    # define LSTM
    input_img = Input(shape=(None, row, col, 1), name='input')
    x = TimeDistributed(Conv2D(16, (2, 2), activation='relu'))(input_img)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
    x = Dropout(0.25)(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(75)(x)
    # model.add(Dropout(0.25))
    x = BatchNormalization()(x)

    x = RepeatVector(4)(x)
    x = LSTM(50, return_sequences=True)(x)
    # model.add(Dropout(0.25))
    x = BatchNormalization()(x)
    output = TimeDistributed(Dense(4, activation='softmax'))(x)

    model = Model(input_img, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model;


def modelA(row, col):
    # define LSTM
    input = Input(shape=(None, row, col, 1), name='main_input')
    x = TimeDistributed(Conv2D(16, (2, 2), activation='relu'))(input)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
    x = Dropout(0.25)(x)
    x = TimeDistributed(Flatten())(x)
    lstm_output = LSTM(75)(x)
    lstm_output = BatchNormalization()(lstm_output)

    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_output)
    auxiliary_input = Input(shape=(4,), name='aux_input')
    x = concatenate([lstm_output, auxiliary_input])

    x = RepeatVector(4)(x)
    x = LSTM(50, return_sequences=True)(x)
    # model.add(Dropout(0.25))
    x = BatchNormalization()(x)
    output = TimeDistributed(Dense(4, activation='softmax'), name='main_output')(x)

    model = Model(inputs=[input, auxiliary_input], outputs=[output, auxiliary_output])
    model.compile(loss={'main_output': 'categorical_crossentropy', 'aux_output': 'binary_crossentropy'},
                  loss_weights={'main_output': 1., 'aux_output': 0.2}, optimizer='adam', metrics=['accuracy'])
    return model


# adjust the BatchNormalization
# https://www.dlology.com/blog/how-to-do-real-time-trigger-word-detection-with-keras/
# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
def modelB(row, col, parameter=None):
    # define LSTM
    input = Input(shape=(None, row, col, 1), name='main_input')
    '''    x = TimeDistributed(Conv2D(16, (2, 2)))(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    '''
    # tower_1 = TimeDistributed(Conv2D(16, (1, 1), padding='same', activation='relu'))(input)
    # tower_1 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(tower_1)

    tower_2 = TimeDistributed(Conv2D(16, (1, 1), padding='same'))(input)
    x = BatchNormalization()(tower_2)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    tower_2 = TimeDistributed(Conv2D(16, (5, 5), padding='same'))(x)
    x = BatchNormalization()(tower_2)
    x = Activation('relu')(x)
    tower_2 = Dropout(0.25)(x)

    tower_3 = TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))(input)
    tower_3 = TimeDistributed(Conv2D(16, (1, 1), padding='same'))(tower_3)
    x = BatchNormalization()(tower_3)
    x = Activation('relu')(x)
    tower_3 = Dropout(0.25)(x)
    concatenate_output = concatenate([tower_2, tower_3], axis=-1)

    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(concatenate_output)
    x = Dropout(0.25)(x)
    x = TimeDistributed(Flatten())(x)
    # convLstm = ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=False)(x)
    lstm_output = LSTM(75)(x)
    lstm_output = BatchNormalization()(lstm_output)
    # lstm_output = BatchNormalization()(convLstm)
    # auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_output)
    # auxiliary_input = Input(shape=(4,), name='aux_input')
    # x = concatenate([lstm_output, auxiliary_input])

    x = RepeatVector(4)(lstm_output)
    x = LSTM(50, return_sequences=True)(x)
    # model.add(Dropout(0.25))
    x = BatchNormalization()(x)
    output = TimeDistributed(Dense(4, activation='softmax'), name='main_output')(x)

    model = Model(inputs=[input], outputs=[output])
    model.compile(loss={'main_output': 'categorical_crossentropy'},
                  loss_weights={'main_output': 1.}, optimizer='adam', metrics=['accuracy'])
    return model


def modelC(row, col):
    # define LSTM
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (2, 2), activation='relu'), input_shape=(None, row, col, 1)))
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


def modelDemoStandard(row, col):
    # define LSTM
    input = Input(shape=(None, row, col, 1), name='main_input')
    x = TimeDistributed(Conv2D(16, (2, 2), activation='relu'))(input)

    x = TimeDistributed(Flatten())(x)
    lstm_output = LSTM(75)(x)

    x = RepeatVector(4)(lstm_output)
    x = LSTM(50, return_sequences=True)(x)

    output = TimeDistributed(Dense(4, activation='softmax'), name='main_output')(x)

    model = Model(inputs=[input], outputs=[output])
    model.compile(loss={'main_output': 'categorical_crossentropy'},
                  loss_weights={'main_output': 1.}, optimizer='adam', metrics=['accuracy'])
    return model


def modelDemoStandardConvLSTM(row, col):
    # define LSTM
    input = Input(shape=(None, row, col, 1), name='main_input')
    # x = TimeDistributed(Flatten())(x)
    x = ConvLSTM2D(filters=75, kernel_size=(3, 3), padding='same', return_sequences=False)(input)
    x = (Flatten())(x)

    x = RepeatVector(4)(x)
    x = LSTM(50, return_sequences=True)(x)

    output = TimeDistributed(Dense(4, activation='softmax'), name='main_output')(x)

    model = Model(inputs=[input], outputs=[output])
    model.compile(loss={'main_output': 'categorical_crossentropy'},
                  loss_weights={'main_output': 1.}, optimizer='adam', metrics=['accuracy'])
    return model


def modelDemoStandardConvLSTMInception(row, col):
    # define LSTM
    input = Input(shape=(None, row, col, 1), name='main_input')

    I_1 = TimeDistributed(Conv2D(16, (1, 1), activation='relu', padding='same', name='C_1'), name='I_11')(input)
    I_1 = TimeDistributed(Conv2D(16, (5, 5), activation='relu', padding='same', name='C_2'), name='I_12')(I_1)

    I_2 = TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='C_3'), name='I_21')(input)
    I_2 = TimeDistributed(Conv2D(16, (1, 1), activation='relu', padding='same', name='C_4'), name='I_22')(I_2)

    concatenate_output = concatenate([I_1, I_2], axis=-1)

    # x = TimeDistributed(Flatten())(x)
    x = ConvLSTM2D(filters=75, kernel_size=(3, 3), padding='same', return_sequences=False)(concatenate_output)
    x = (Flatten())(x)

    x = RepeatVector(4)(x)
    x = LSTM(50, return_sequences=True)(x)

    output = TimeDistributed(Dense(4, activation='softmax'), name='main_output')(x)

    model = Model(inputs=[input], outputs=[output])
    model.compile(loss={'main_output': 'categorical_crossentropy'},
                  loss_weights={'main_output': 1.}, optimizer='adam', metrics=['accuracy'])
    return model


def ModelShare():
    tweet_a = Input(shape=(280, 256))
    tweet_b = Input(shape=(280, 256))

    # This layer can take as input a matrix
    # and will return a vector of size 64
    shared_lstm = LSTM(64, return_sequences=True, name='lstm')

    # When we reuse the same layer instance
    # multiple times, the weights of the layer
    # are also being reused
    # (it is effectively *the same* layer)
    encoded_a = shared_lstm(tweet_a)
    encoded_b = shared_lstm(tweet_b)

    # We can then concatenate the two vectors:
    merged_vector = concatenate([encoded_a, encoded_b], axis=-1)

    # And add a logistic regression on top
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    # We define a trainable model linking the
    # tweet inputs to the predictions
    model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def ModelInception():
    input_img = Input(shape=(256, 256, 3))

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=1)

    model = Model(inputs=[input_img], outputs=output)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def ModelResidual():
    # input tensor for a 3-channel 256x256 image
    x = Input(shape=(256, 256, 3))
    # 3x3 conv with 3 output channels (same as input channels)
    y = Conv2D(3, (3, 3), padding='same')(x)
    # this returns x + y.
    z = add([x, y])

    model = Model(inputs=[x], outputs=z)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def ModelSharedVision():
    # First, define the vision modules
    digit_input = Input(shape=(27, 27, 1))
    x = Conv2D(64, (3, 3))(digit_input)
    x = Conv2D(64, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    out = Flatten()(x)

    vision_model = Model(digit_input, out)

    # Then define the tell-digits-apart model
    digit_a = Input(shape=(27, 27, 1))
    digit_b = Input(shape=(27, 27, 1))

    # The vision model will be shared, weights and all
    out_a = vision_model(digit_a)
    out_b = vision_model(digit_b)

    concatenated = concatenate([out_a, out_b])
    out = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model([digit_a, digit_b], out)
    return classification_model


def ModelVisualQuestionAnswering():
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = Sequential()
    vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    vision_model.add(Conv2D(64, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    vision_model.add(Conv2D(128, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    # Now let's get a tensor with the output of our vision model:
    image_input = Input(shape=(224, 224, 3))
    encoded_image = vision_model(image_input)

    # Next, let's define a language model to encode the question into a vector.
    # Each question will be at most 100 words long,
    # and we will index words as integers from 1 to 9999.
    question_input = Input(shape=(100,), dtype='int32')
    embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
    encoded_question = LSTM(256)(embedded_question)

    # Let's concatenate the question vector and the image vector:
    merged = concatenate([encoded_question, encoded_image])

    # And let's train a logistic regression over 1000 words on top:
    output = Dense(1000, activation='softmax')(merged)

    # This is our final model:
    vqa_model = Model(inputs=[image_input, question_input], outputs=output)
    return vqa_model


def ModelVideoQuestionAnswering():
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = Sequential()
    vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    vision_model.add(Conv2D(64, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    vision_model.add(Conv2D(128, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    # Now let's get a tensor with the output of our vision model:
    image_input = Input(shape=(224, 224, 3))
    encoded_image = vision_model(image_input)

    # Next, let's define a language model to encode the question into a vector.
    # Each question will be at most 100 words long,
    # and we will index words as integers from 1 to 9999.
    question_input = Input(shape=(100,), dtype='int32')
    embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
    encoded_question = LSTM(256)(embedded_question)

    # Let's concatenate the question vector and the image vector:
    merged = concatenate([encoded_question, encoded_image])

    # And let's train a logistic regression over 1000 words on top:
    output = Dense(1000, activation='softmax')(merged)

    # This is our final model:
    # vqa_model = Model(inputs=[image_input, question_input], outputs=output)

    video_input = Input(shape=(100, 224, 224, 3))
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
    encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

    # This is a model-level representation of the question encoder, reusing the same weights as before:
    question_encoder = Model(inputs=question_input, outputs=encoded_question)

    # Let's use it to encode the question:
    video_question_input = Input(shape=(100,), dtype='int32')
    encoded_video_question = question_encoder(video_question_input)

    # And this is our video question answering model:
    merged = concatenate([encoded_video, encoded_video_question])
    output = Dense(1000, activation='softmax')(merged)
    video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)

    return video_qa_model
