import keras
import numpy as np
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import gc
import pickle


def load_cpickle_gc(file):
    output = open(file, 'rb')
    gc.disable()
    pkl = pickle.load(output)
    gc.enable()
    output.close()
    mydict = np.array(pkl).reshape(15250,320,5)
    return mydict


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(64,64,64), n_channels=1,
                 n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = 3
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def correct_dim_cnn(array):
        output=[]
        for i in array:
            print(np.shape(i))
            output.append(i.reshape(1, 15250, 320, 5))
        return output


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = load_cpickle_gc(ID)
            # Store class
            y[i] = self.labels[ID]
#        output=[]
#        for i in X:
#            output.append(i.reshape(15250, 320, 5))       
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

