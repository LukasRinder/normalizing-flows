'''
Implementation of functions that are important for training.

List of functions:
1. sanity_check
'''


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


'''---------------------------------------- Negative Log Likelihood -------------------------------------------------'''

@tf.function
def nll(distribution, data):
    """
    Computes the negative log liklihood loss for a given distribution and given data.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param data: Data or a batch from data.
    :return: Negative Log Likelihodd loss.
    """
    return -tf.reduce_mean(distribution.log_prob(data))


'''--------------------------------------------- Train function -----------------------------------------------------'''

@tf.function
def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))

    return loss


def train_density_no_tf(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows without tf.function decorator
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(distribution.log_prob(batch)) # negative log likelihood
        gradients = tape.gradient(loss, distribution.trainable_variables)
        optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))
        return loss
        

'''----------------- Sanity check: after training the integral of the pdf has to sum up to one ----------------------'''


def sanity_check(dist, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=1000):
    '''
    Implementation of a approximated integral over a mesh grid from [xmin, xmax, ymin, ymax].
    The higher mesh_count, the more accurate the approximation.

    :param dist: Tensorflow distribution (tfp.distribution).
    :param xmin: Min x value of mesh grid (float32).
    :param xmax: Max x value of mesh grid (float32).
    :param ymin: Min y value of mesh grid (float32).
    :param ymax: Max y value of mesh grid (float32).
    :param mesh_count: Number of samples per axis of the mesh_grid (int).
    :return: Approximated integral of dist over a mesh grid (should be close to 1).
    '''

    # create 2D mesh grid with mesh_count samples
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)

    # concatenate the coordinates in an array
    concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))

    # calculate dA for the integral
    dA = ((xmax - xmin) * (ymax - ymin)) / (mesh_count ** 2)

    # calculate the probabilities of the concatenated samples and return the approximated integral
    pm = dist.prob(concatenated_mesh_coordinates)

    return tf.reduce_sum(pm) * dA


'''------------------------------------ Train-Validation-Test Split -------------------------------------------------'''


def shuffle_split(samples, train_split, val_split):
    '''
    Shuffles the data and performs a train-validation-test split.
    Test = 1 - (train + val).
    
    :param samples: Samples from a dataset / data distribution.
    :param train: Portion of the samples used for training (float32, 0<=train<1).
    :param val: Portion of the samples used for validation (float32, 0<=val<1).
    :return train_data, val_data, test_data: 
    '''

    if train_split + val_split > 1:
        raise Exception('train_split plus val_split has to be smaller or equal to one.')

    batch_size = len(samples)
    np.random.shuffle(samples)
    n_train = int(round(train_split * batch_size))
    n_val = int(round((train_split + val_split) * batch_size))
    train_data = tf.cast(samples[0:n_train], dtype=tf.float32)
    val_data = tf.cast(samples[n_train:n_val], dtype=tf.float32)
    test_data = tf.cast(samples[n_val:batch_size], dtype=tf.float32)

    return train_data, val_data, test_data


def checkerboard(height, width, reverse=False, dtype=tf.float32):
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)] 
    checkerboard = tf.convert_to_tensor(checkerboard, dtype = dtype)
    if reverse:
        checkerboard = 1 - checkerboard
    
    checkerboard = tf.reshape(checkerboard, (1,height,width,1))
        
    return tf.cast(checkerboard, dtype=dtype)