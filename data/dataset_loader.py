"""
Functions to import and preprocess various datasets.
Currently implemented datasets:

1. MNIST
2. Celeb A
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfkd = tf.keras.datasets

from data import uci_classes

"""----------------------------- Preprocessing Functions --------------------------------"""


def logit(z, beta=10e-6):
    """
    Conversion to logit space according to equation (24) in [Papamakarios et al. (2017)].
    Includes scaling the input image to [0, 1] and conversion to logit space.
    :param z: Input tensor, e.g. image. Type: tf.float32.
    :param beta: Small value. Default: 10e-6.
    :return: Input tensor in logit space.
    """

    inter = beta + (1 - 2 * beta) * (z / 256)
    return tf.math.log(inter/(1-inter))  # logit function


def inverse_logit(x, beta=10e-6):
    """
    Reverts the preprocessing steps and conversion to logit space and outputs an image in
    range [0, 256]. Inverse of equation (24) in [Papamakarios et al. (2017)].
    :param x: Input tensor in logit space. Type: tf.float32.
    :param beta: Small value. Default: 10e-6.
    :return: Input tensor in logit space.
    """

    x = tf.math.sigmoid(x)
    return (x-beta)*256 / (1 - 2*beta)


"""-------------------------------------- MNIST -----------------------------------------"""


def load_and_preprocess_mnist(logit_space=True, batch_size=128, shuffle=True, classes=-1, channels=False):
    """
     Loads and preprocesses the MNIST dataset. Train set: 50000, val set: 10000,
     test set: 10000.
    :param logit_space: If True, the data is converted to logit space.
    :param batch_size: batch size
    :param shuffle: bool. If True, dataset will be shuffled.
    :param classes: int of class to take, defaults to -1 = ALL
    :return: Three batched TensorFlow datasets:
    batched_train_data, batched_val_data, batched_test_data.
    """

    (x_train, y_train), (x_test, y_test) = tfkd.mnist.load_data()

    # reserve last 10000 training samples as validation set
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # if logit_space: convert to logit space, else: scale to [0, 1]
    if logit_space:
        x_train = logit(tf.cast(x_train, tf.float32))
        x_test = logit(tf.cast(x_test, tf.float32))
        x_val = logit(tf.cast(x_val, tf.float32))
        interval = 256
    else:
        x_train = tf.cast(x_train / 256, tf.float32)
        x_test = tf.cast(x_test / 256, tf.float32)
        x_val = tf.cast(x_val / 256, tf.float32)
        interval = 1


    if classes == -1:
        pass
    else:
        #TODO: Extract Multiple classes: How to to the train,val split,
        # Do we need to to a class balance???
        x_train = np.take(x_train, tf.where(y_train == classes), axis=0)
        x_val = np.take(x_val, tf.where(y_val == classes), axis=0)
        x_test = np.take(x_test, tf.where(y_test == classes), axis=0)

    # reshape if necessary
    if channels:
        x_train = tf.reshape(x_train, (x_train.shape[0], 28, 28, 1))
        x_val = tf.reshape(x_val, (x_val.shape[0], 28, 28, 1))
        x_test = tf.reshape(x_test, (x_test.shape[0], 28, 28, 1))
    else:
        x_train = tf.reshape(x_train, (x_train.shape[0], 28, 28))
        x_val = tf.reshape(x_val, (x_val.shape[0], 28, 28))
        x_test = tf.reshape(x_test, (x_test.shape[0], 28, 28))

    if shuffle:
        shuffled_train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)

    batched_train_data = shuffled_train_data.batch(batch_size)
    batched_val_data = tf.data.Dataset.from_tensor_slices(x_val).batch(batch_size)
    batched_test_data = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)    
    
    return batched_train_data, batched_val_data, batched_test_data, interval


"""-------------------------------------------- UCI datasets --------------------------------------------------------"""


def load_and_preprocess_uci(uci_dataset="power", batch_size=128, shuffle=True):
    """
    Loads and preprocesses the uci dataset. See more details in uci_classes.
    Downdload the datasets from: https://zenodo.org/record/1161203#.Wmtf_XVl8eN
    To do so run:
    wget https://zenodo.org/record/1161203/files/data.tar.gz
    
    Make a directory uci_data in the data directory.
    Unpack the downloaded file, and place it in the uci_data directory.
    
    :param uci_dataset: name of the uci dataset. (power, gas, miniboone, hepmass)
    :param batch_size:
    :param shuffle:
    :return: Three tuples of tf Tensors.

    """
    if uci_dataset == "power":
        data = uci_classes.POWER()
    elif uci_dataset == "gas":
        data = uci_classes.GAS()
    elif uci_dataset == "miniboone":
        data = uci_classes.MINIBOONE()
    elif uci_dataset == "hepmass":
        raise ValueError("Not implemented")
        #data = uci_classes.HEPMASS()
    else:
        raise ValueError("Dataset not known.")
    
    maxes = np.max(data.trn.x, axis=0)
    mins = np.min(data.trn.x, axis=0)
    intervals = np.ceil(np.maximum(np.absolute(maxes), np.absolute(mins)))  
    
    data_train = tf.data.Dataset.from_tensor_slices(data.trn.x)
    data_validate = tf.data.Dataset.from_tensor_slices(data.val.x)
    data_test = tf.data.Dataset.from_tensor_slices(data.tst.x)
    
    if shuffle:
        data_train = data_train.shuffle(10000)

    data_train = data_train.batch(batch_size)
    data_validate = data_validate.batch(batch_size)
    data_test = data_test.batch(batch_size)
    # maybe need to batch val and test as well. Not sure yet.

    return data_train, data_validate, data_test, intervals
    
def load_and_preprocess_celeb(batch_size=32, shuffle=True, download=True):
    # get preprocessed train, validation, and test data
    celeb_dataset = tfds.load(name="celeb_a", batch_size=batch_size, shuffle_files=shuffle, download=download)
    batched_train_data = celeb_dataset["train"]
    batched_val_data = celeb_dataset["validation"]
    batched_test_data = celeb_dataset["test"]
    
    return data_train, data_validate, data_test
