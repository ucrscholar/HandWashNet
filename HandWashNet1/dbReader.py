import cv2
import numpy as np
import tensorflow.keras
from PIL import Image
from numpy import asarray


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,samples=1,frames=100, batch_size=32, dim=(32,32,32), n_channels=3,
                 n_classes=8, shuffle=False):
        'Initialization'
        self.dim = dim
        self.samples = samples
        self.frames = frames
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs))*14)
        #return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        k = int(index/14)
        samples= self.list_IDs[k]
        IDs = index%14 *4

        # Accessing a text file - www.101computing.net/mp3-playlist/
        fileNameTrain = "D:/data1/" + samples + '.dat'
        file = open(fileNameTrain, "r")

        # Repeat for each song in the text file
        fields=[]
        for line in file:
            # Let's split the line into an array called "fields" using the ";" as a separator:
            field = line.split(",")
            for i in range(0,28,4):
                fields.append('BKA')
                fields.append('0' if i==0 else field[i-1])
                fields.append('BKB')
                fields.append(field[i+1])
                fields.append(field[i])
                fields.append(field[i+1])
                fields.append(field[i+2])
                fields.append(field[i+3])


        # It is good practice to close the file at the end to free up resources
        file.close()


        #list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #label =
        # Generate data
        X, y = self.__data_generation(samples, fields, IDs)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, samples, fields, IDs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization


        # Generate data
        NameA =fields[IDs]
        va = np.long(fields[IDs + 1])
        NameB = fields[IDs+2]
        vb = np.long(fields[IDs+3])
        Name = NameA[0:2]

        X = np.empty((self.samples, vb-va, *self.dim, self.n_channels))
        y = np.empty((self.samples), dtype=int)
        for i in range(0,vb-va):
            # load the image
            fileName = "D:/data1/" + samples + '/_Color_'+str(va+i)+'.png'
            image = cv2.imread(fileName)
            # convert image to numpy array
            # Store sample
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.resize(gray_image,(50,50))
            data = np.expand_dims(gray_image2, axis=-1)
            X[0,i,] = data


        # Store class
        y[0] = self.labels[Name]

        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)