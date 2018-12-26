import keras
import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil
from glob import glob

COLORS = {
    'green': ['\033[32m', '\033[39m'],
    'red': ['\033[31m', '\033[39m']
}


def get_best_model_ckpt(model_path):
    list_best_model = glob(model_path+'*.index')
    if len(list_best_model):
        model = list_best_model[0].split('.index')[0]
        return model
    else:
        return None


def get_generators(batch_size=16, data_format="channels_last", verbose=True):
    print("... loading CIFAR10 dataset ...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    if data_format == "channels_last" and x_train.shape[-1] != 3:
        shape = x_train.shape[1:]
        x_train = np.transpose(x_train, axes=[0, 2, 3, 1])
        x_test = np.transpose(x_test, axes=[0, 2, 3, 1])
        if verbose: print("\ttranspose images from {} to {}".format(shape, x_train.shape[1:]))

    if data_format == "channels_first" and x_train.shape[1] != 3:
        shape = x_train.shape[1:]
        x_train = np.transpose(x_train, axes=[0, 3, 1, 2])
        x_test = np.transpose(x_test, axes=[0, 3, 1, 2])
        if verbose: print("\ttranspose images from {} to {}".format(shape, x_train.shape[1:]))

    # remove unused last dim [N, 1] -> [N,]
    y_train = np.squeeze(y_train).astype(np.int32)
    y_test = np.squeeze(y_test).astype(np.int32)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      stratify=y_train,
                                                      shuffle=True,
                                                      random_state=51)
    # cast samples and labels
    # perform one hot encoding
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    if verbose:
        print("\tTRAIN - images {} | {} - labels {} - {}".format(x_train.shape, x_train.dtype, y_train.shape, y_train.dtype))
        print("\tVAL - images {} | {} - labels {} - {}".format(x_val.shape, x_val.dtype, y_val.shape, y_val.dtype))
        print("\tTEST - images {} | {} - labels {} - {}\n".format(x_test.shape, x_test.dtype, y_test.shape, y_test.dtype))

    generator_aug = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                                                                 samplewise_std_normalization=True,
                                                                 width_shift_range=5,
                                                                 height_shift_range=5,
                                                                 fill_mode='constant',
                                                                 cval=0.0,
                                                                 horizontal_flip=True,
                                                                 vertical_flip=False,
                                                                 data_format=data_format)

    generator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                                                             samplewise_std_normalization=True,
                                                             data_format=data_format)

    # python iterator object that yields augmented samples
    iterator_train = generator_aug.flow(x_train, y_train, batch_size=batch_size)

    # python iterators object that yields not augmented samples
    iterator_valid = generator.flow(x_val, y_val, batch_size=batch_size)
    iterator_test = generator.flow(x_test, y_test, batch_size=batch_size)

    steps_per_epoch_train = int(ceil(iterator_train.n / batch_size))
    steps_per_epoch_val = int(ceil(iterator_valid.n / batch_size))
    steps_per_epoch_test = int(ceil(iterator_test.n / batch_size))

    if verbose:
        print("\tsteps per epoch - TRAIN : ", steps_per_epoch_train)
        print("\tsteps per epoch - VALID : ", steps_per_epoch_val)
        print("\tsteps per epoch - TEST : ", steps_per_epoch_test)

    return iterator_train, iterator_valid, iterator_test, \
        steps_per_epoch_train, steps_per_epoch_val, steps_per_epoch_test