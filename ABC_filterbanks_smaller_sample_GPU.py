from blimpy.io.file_wrapper import FilReader as F
from blimpy.io.sigproc import read_header
import numpy as np
import pandas as pd
import argparse
import random
import mmap
from multiprocessing import Pool
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
import glob
import gc  
from sklearn.utils import shuffle
import tensorflow as tf
from my_classes import DataGenerator
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from tensorflow.python.keras.utils.data_utils import Sequence

def header_size(file):
    filfile=open(file,'rb')
    filfile.seek(0)
    round1 = filfile.read(1000)
    headersize = round1.find('HEADER_END'.encode('utf-8'))+len('HEADER_END'.encode('utf-8'))
    return headersize

def normalise(data):
    data = np.array(data, dtype=np.float32)
    data -= np.median(data)
    data /= np.std(data)
    return data


def get_data(tstart, tstop, file):
    print(file)
    header = F(file, load_data = False).header
    nifs = header[b'nifs']
    nchans = header[b'nchans']
    filfile = open(file,'rb')
    bstart = (int(tstart) * nchans) + header_size(file)
    nbytes = (int(tstop - tstart) * nchans)
    mmdata = mmap.mmap(filfile.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
    return np.frombuffer(mmdata[int(bstart):int(nbytes+bstart)], dtype='uint8').reshape((-1,nifs,nchans))

def concatenate(files_for_concat, tstart, tstop):
    concatenated_array = []
    for i in files_for_concat:
        data = normalise(get_data(tstart, tstop, i)[:, 0, :])
        concatenated_array.append(data)
    return concatenated_array

def main(args):
    file_main, index = args[0], args[1]
    beam = file_main.split("/")[-1].split("_")[3]
    isamp = file_main.split("/")[-1].split("_")[4]
    fur_name = file_main.split("furby")[1].split("_")[1]
    file_side = file_main.split("injected_main_beam")[0] +'injected_side_beam_' + str(beam) + '_' + str(isamp) + '_' + 'furby' + '_' + fur_name + '_BEAM_' + str(int(float(beam)) + 1).zfill(3) + '_2020-05-08-07:33:43.fil'
    list_furby = []
    list_rfi = []
    for i in range(8):
        if i == 3:
            list_rfi.append("/fred/oz002/users/VG/FRB200508/Hires_filterbanks/FRB20200508/BEAM_" + str(int(float(beam)) -3 + i).zfill(3)  + "/2020-05-08-07:33:43.fil")
            list_furby.append(file_main)
        elif i == 4:
            list_rfi.append("/fred/oz002/users/VG/FRB200508/Hires_filterbanks/FRB20200508/BEAM_" + str(int(float(beam)) -3 + i).zfill(3)  + "/2020-05-08-07:33:43.fil")
            list_furby.append(file_side)
        else:
            high_res = "/fred/oz002/users/VG/FRB200508/Hires_filterbanks/FRB20200508/BEAM_" + str(int(float(beam)) -3 + i).zfill(3)  + "/2020-05-08-07:33:43.fil"
            list_furby.append(high_res)
            list_rfi.append(high_res)
    a = random.randint(0,3)
    files_for_concat_furby = list_furby[a:a+5]
    files_for_concat_rfi  = list_rfi[a:a+5]
    tstart = int(isamp) - 7625
    tstop = int(isamp) + 7625
    X_fur[index] = concatenate(files_for_concat_furby, tstart, tstop)
    X_rfi[index] = concatenate(files_for_concat_rfi, tstart, tstop)
    y_fur[index] = 1
    y_rfi[index] = 0




#def BatchGenerator(pickle_file_path, batch_size):
#    RFI_files = glob.glob(pickle_file_path + '*rfi.pkl')
#    Furby_files = glob.glob(pickle_file_path + '*.fur.pkl')
#
#    X_furby=[]
#    X_rfi=[]
#    for i,n in enumerate(RFI_files):
#        if i==0:
#            X_rfi = load_cpickle_gc(n)[0]
#        else:
#            print(np.shape(X_rfi))
#            X_rfi = np.concatenate((X_rfi, load_cpickle_gc(n)[0]), axis = 0)
#
#    for i,n in enumerate(Furby_files):
#        if i==0:
#            X_furby = load_cpickle_gc(n)[0]
#        else:
#            X_furby = np.concatenate((X_furby, load_cpickle_gc(n)[0]), axis = 0)
#
#    X = np.concatenate([X_fur, X_rfi], axis = 0)
#    y = np.concatenate([np.ones(np.shape(X_rfi)[0], dtype=np.uint8), np.zeros(np.shape(X_rfi)[0], dtype=np.uint8)], axis = 0)
#    final = [X, y]
#    return final

def cnn_model(pickle_file_path):
    print("Reached cnn model")
    params = {'dim': (15250, 320, 5),
            'batch_size': 10,
            'n_classes': 2,
            'n_channels': 1,
            'shuffle': True} 
    RFI_files = glob.glob(pickle_file_path + '*rfi_*.pkl')
    Furby_files = glob.glob(pickle_file_path + '*fur*.pkl')
    train = []
    validation = []

    labels = []
    train = RFI_files[:-483]
    print(np.shape(train))
    train = np.concatenate((train, Furby_files[:-483]), axis = 0)
    print(np.shape(train))
    validation = RFI_files[-483:]
    print(np.shape(validation))
    validation = np.concatenate((validation, Furby_files[-483:]), axis = 0)
    print(np.shape(validation))
    labels_RFI = np.zeros(np.shape(RFI_files))
    labels_fur = np.ones(np.shape(Furby_files))
    labels = np.concatenate((labels_RFI, labels_fur), axis = 0)
#    print(np.concatenate((RFI_files, Furby_files), axis = 0)[0])


    partition = {'train': train, 'validation': validation}
    label = dict(zip(np.concatenate((RFI_files, Furby_files), axis = 0), labels))
#    print(label)
    print(np.shape(label))
    print(np.shape(partition['train']))
    print(np.shape(partition['validation']))

    # Generators
    training_generator = DataGenerator(partition['train'], label, **params)
    validation_generator = DataGenerator(partition['validation'], label, **params)
 
#    le = LabelEncoder()
#    yy = to_categorical(le.fit_transform(y))

#    X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

#    print("split the data into training and test sets")

 #   def correct_dim_cnn(array):
 #       output=[]
 #       for i in array:
 #           print(np.shape(i))
 #           output.append(i.reshape(15250, 320, 5))
 #       return output

 #   X_train = correct_dim_cnn(X_train)
 #   X_test = correct_dim_cnn(X_test)

 #   print(np.shape(X_train))
 #   print(np.shape(X_test))
 #   print(y_test)
 #   print(y_train)



    def create_cnn(height, width, depth, filters=(250, 32, 5), regress=False):
        HWD = (height, width, depth)
        print(HWD)
        Dim = -1
        inputs = Input(shape = HWD)
        for (i, f) in enumerate(filters):
            if i == 0:
                x = inputs
            x = Conv2D(f, (3, 3), padding="valid")(x)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = BatchNormalization(axis=Dim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=Dim)(x)
        x = Dropout(0.5)(x)
        x = Dense(4)(x)
        x = Activation("relu")(x)
        model = Model(inputs, x)
        return model

    cnn_model = create_cnn(15250, 320, 5, regress=False)
    cnn_model.summary()
    num_labels = 2
    x = Dense(4, activation="relu")(cnn_model.output)
    x = Dense(num_labels, activation="softmax")(x)
    model = Model(inputs=[cnn_model.input], outputs=x)
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)


    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
    model.summary()
    num_epochs = 100
    num_batch_size = 10
    checkpointer = ModelCheckpoint(filepath= pickle_file_path +'weights.best.ABC_returns_learning_rate_0.001_batch_150_epochs_100.hdf5', verbose=1, save_best_only=True)
    start = datetime.now()

#    with tf.compat.v1.Session( config = tf.compat.v1.ConfigProto( log_device_placement = True ) ):
    model.fit_generator(generator=training_generator, validation_data=validation_generator)

#    model.fit_generator(generator=training_generator,
#            validation_data=validation_generator,
#            use_multiprocessing=True,
#            workers=6, callbacks=[checkpointer], verbose=1)
#    model.fit_generator(generator=training_generator, validation_data=validation_generator)
    
    #model.fit([X_train], y_train, validation_data = ([X_test], y_test), batch_size=num_batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    score = model.evaluate([x_freq_time_test], y_test, verbose=1)
    accuracy = 100*score
    print("Accuracy on validation set: %.4f%%" % accuracy)
    model_json = model.to_json()
    with open(pickle_file_path + "model_ABC_returns_learning_rate_0.001_batch_150_epochs_100.json", "w") as json_file:
        json_file.write(model_json)


def load_cpickle_gc(file): 
    output = open(file, 'rb')
    gc.disable()
    mydict = pickle.load(output)
    gc.enable() 
    output.close() 
    return mydict

if __name__ == '__main__':
    a = argparse.ArgumentParser()
#    a.add_argument('-c', '--cand_param_file', help='csv file with candidate parameters', type=str)
#    a.add_argument('-n', '--nproc', help='number of processors', type=int, default = 2)
    a.add_argument('-p', '--pickle_file', help='pickle file of arrays', type=str, default=None)
    a.add_argument('-o', '--output_dir', help='output directory', type=str)
    values = a.parse_args()
#    RFI_files = glob.glob(values.pickle_file + '*rfi.pkl')
#    Furby_files = glob.glob(values.pickle_file + '*fur.pkl')
#
#    X_furby=[]
#    X_rfi=[]
#    for i,n in enumerate(RFI_files):
#        if i==0:
#            X_rfi = load_cpickle_gc(n)[0]
#        else:
#            print(np.shape(X_rfi))
#            X_rfi = np.concatenate((X_rfi, load_cpickle_gc(n)[0]), axis = 0)
#
#    print(np.shape(X_rfi))
#    for i,n in enumerate(Furby_files):
#        if i==0:
#            X_furby = load_cpickle_gc(n)[0]
#        else:
#            X_furby = np.concatenate((X_furby, load_cpickle_gc(n)[0]), axis = 0)
#
#    print(np.shape(X_furby))
#
#    X = np.concatenate((X_furby, X_rfi), axis = 0)
#    y = np.concatenate((np.ones(np.shape(X_furby)[0], dtype=np.uint8), np.zeros(np.shape(X_rfi)[0], dtype=np.uint8)), axis = 0)

#    final = [X, y]
#    print(np.shape(final))

    cnn_model(values.pickle_file)
    exit()

    if values.pickle_file == None:
        df = pd.read_csv(values.cand_param_file, header = None)
        X_fur = np.zeros((df.shape[0], 5, 15250, 320), dtype=np.uint8)
        X_rfi = np.zeros((df.shape[0], 5, 15250, 320), dtype=np.uint8)
        y_fur = np.zeros(df.shape[0], dtype=np.uint8)
        y_rfi = np.zeros(df.shape[0], dtype=np.uint8)
        files_index = []
        for i, r in df.iterrows():
            main([r[0], i])
        final_fur = [X_fur, y_fur]
        final_rfi = [X_rfi, y_rfi]
        with open(values.cand_param_file.split(".csv")[0] + 'concatenated_arrays_during_training_smaller_sample_fur.pkl','wb') as f:
            pickle.dump(final_fur, f)

        with open(values.cand_param_file.split(".csv")[0] + 'concatenated_arrays_during_training_smaller_sample_rfi.pkl','wb') as f:
            pickle.dump(final_rfi, f)
        exit()

        X = np.concatenate([X_fur, X_rfi], axis = 0)
        y = np.concatenate([y_fur, y_rfi], axis = 0)
        print(y)
        final = [X,y]
        print(np.shape(final[0]), np.shape(final[1]))
        with open(values.cand_param_file.split(".csv")[0] + 'concatenated_arrays_during_training_smaller_sample.pkl','wb') as f:
            pickle.dump(final, f)
            print(final[1])
    else:
        with open(values.pickle_file,'rb') as f:
            final = pickle.load(f)
            print(final[1])
            print(np.shape(final[0]), np.shape(final[1]))

#    cnn_model(final[0], final[1])












