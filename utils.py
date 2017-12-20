import numpy as np
np.random.seed(12345678) # for reproducibility

from keras import regularizers, losses, callbacks
from keras import backend as K
from math import sqrt

import tensorflow as tf

def KLregularizer(expected, coeff, tanh = True):
    """Kullback-Leibler regularizer for sparse autoencoder"""

    # tanh scaling to [0,1] interval
    if tanh:
        expected = (1 + expected) / 2
    
    def kl(x):
        """
        The Kullback-Leibler divergence for two Bernoulli distributions
        with mean `expected` and `x` respectively
        """

        # tanh scaling to [0,1] interval
        if tanh:
            x = (1 + x) / 2
        
        return tf.cond(
            # if x <= 0 or x == 1
            tf.logical_or(tf.less_equal(expected / x, 0), tf.less_equal((1 - expected)/(1 - x), 0)),
            # just return 0
            lambda: 0.,
            # else, compute KL divergence
            lambda: expected * K.log(expected / x) + (1 - expected) * K.log((1 - expected)/(1 - x))
        )

    def reg(observed_activations):
        """
        Calculates the regularization that needs to be applied to improve sparsity
        """
        
        # outputs of encoding layer will be shaped (batch_size, 32)
        # print(observed_activations.get_shape().as_list())
        observed = K.mean(observed_activations, axis = [0])
        # print("shape means: {}".format(observed.get_shape().as_list()))
        
        allkl = K.map_fn(kl, observed)
        sumkl = K.sum(allkl)

        return coeff * sumkl

    return reg

# following https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
def contractive_loss(model, rec_err = losses.binary_crossentropy, tanh = True, lam = 2e-4):
    # derive either tanh or sigmoid
    der_act = (lambda h: 1 - h * h) if tanh else (lambda: h * (1 - h))
    
    def loss(y_pred, y_true):
        rec = rec_err(y_pred, y_true)

        W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer('encoded').output
        dh = der_act(h)  # N_batch x N_hidden
        
        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
        
        return rec + contractive

    return loss

def noise_input(x, proportion = 0.05):
    x_noisy = x + np.around(np.random.uniform(low = 0., high = 0.5 / (1 - proportion), size = x.shape))
    x_noisy = np.clip(x_noisy, 0., 1.)
    
    return x_noisy

def correntropy_loss(sigma = 0.2):
    def robust_kernel(alpha):
        return 1. / (sqrt(2 * np.pi) * sigma) * K.exp(- K.square(alpha) / (2 * sigma * sigma))

    def loss(y_pred, y_true):
        return -K.sum(robust_kernel(y_pred - y_true))

    return loss

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
