#!/usr/bin/env python

import numpy as np
np.random.seed(12345678) # for reproducibility

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import os
os.environ['PYTHONHASHSEED'] = '0'

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers, losses, callbacks
from keras import backend as K
from math import sqrt

import tensorflow as tf

def KLregularizer(expected, coeff):
    """Kullback-Leibler regularizer for sparse autoencoder"""

    # tanh scaling to [0,1] interval
    expected = (1 + expected) / 2
    
    def kl(x):
        """
        The Kullback-Leibler divergence for two Bernoulli distributions
        with mean `expected` and `x` respectively
        """

        # tanh scaling to [0,1] interval
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

# following https://blog.keras.io/building-autoencoders-in-keras.html
class Autoencoder():
    def __init__(self, encoding_dim = 32, weight_decay = True, sparse = False, contractive = False, denoising = False, robust = False, activation = "tanh"):
        self.weight_decay = weight_decay
        self.sparse = sparse
        self.contractive = contractive
        self.denoising = denoising
        self.robust = robust

        self.encoding_dim = encoding_dim
        self.activation = activation

        attrs = []
        if self.sparse: attrs.append("sparse")
        if self.contractive: attrs.append("contractive")
        if self.denoising: attrs.append("denoising")
        if self.robust: attrs.append("robust")
        if not attrs: attrs.append("basic")
        if self.weight_decay: attrs.append("wd")

        self.name = "{}-{}-{}".format(
            "-".join(attrs),
            encoding_dim,
            activation
        )

        self.build()

    def build(self):
        # "encoded" is the encoded representation of the input
        activity_reg = KLregularizer(-0.7, 0.2) if self.sparse else regularizers.l1(0.)
        kernel_reg = regularizers.l2(0.02) if self.weight_decay else regularizers.l1(0.)
        
        # this is our input placeholder
        input_img = Input(shape=(784,))
        
        encoded = Dense(self.encoding_dim, activation=self.activation
                        , activity_regularizer = activity_reg
                        , kernel_regularizer = kernel_reg
                        , name = "encoded")(input_img)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(784, activation=('linear' if self.activation == "linear" else 'sigmoid')
                        , name = "decoded")(encoded)

        # this model maps an input to its reconstruction
        self.model = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.model.layers[-1]
        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        

from keras.datasets import mnist

class Trainer():
    def __init__(self, autoencoder = Autoencoder()):
        self.autoencoder = autoencoder
        (x_train, _), (x_test, self.y_test) = mnist.load_data()
        
        # normalize
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # flatten
        self.x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        self.x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        if autoencoder.denoising:
            self.x_test_noisy = noise_input(self.x_test)

        self.name = ""

    def train(self, optimizer = "rmsprop", loss = losses.binary_crossentropy, epochs = 50):
        # Here, we use binary crossentropy as loss function
        # since the output of our model is in the interval [0,1]
        # and our data is normalized.
        # Otherwise we could use 'mean_squared_error'
        if self.autoencoder.robust:
            loss = correntropy_loss()
            
        if self.autoencoder.contractive:
            loss = contractive_loss(self.autoencoder.model, rec_err = loss)

        self.autoencoder.model.compile(optimizer = optimizer,
                                       loss = loss)

        # train
        history = LossHistory()
        if self.autoencoder.denoising:
            for ep in range(epochs):
                noisy_train = noise_input(self.x_train)
                self.autoencoder.model.fit(noisy_train, self.x_train,
                                           epochs = 1,
                                           batch_size = 256,
                                           shuffle = True,
                                           callbacks=[history],
                                           validation_data = (self.x_test, self.x_test))
        else:
            self.autoencoder.model.fit(self.x_train, self.x_train,
                                       epochs = epochs,
                                       batch_size = 256,
                                       shuffle = True,
                                       callbacks=[history],
                                       validation_data = (self.x_test, self.x_test))

        self.name = "{}-{}".format(
            optimizer,
            "mse" if loss == losses.mean_squared_error else ("xent" if loss == losses.binary_crossentropy or self.autoencoder.contractive else "corr")
        )

        with open("/home/fdavidcl/Documentos/research/publications/2017/ReviewAutoencoders/examples/{}-{}.csv".format(self.autoencoder.name, self.name), "w") as out_file:
            out_file.write(",".join(("{}".format(x) for x in history.losses)))

        return self

    def predict_test(self):
        
        # encode and decode some digits
        # note that we take them from # TODO: he *test* set
        encoded_imgs = self.autoencoder.encoder.predict(self.x_test)
        decoded_imgs = self.autoencoder.decoder.predict(encoded_imgs)

        if self.autoencoder.denoising:
            encoded_noisy = self.autoencoder.encoder.predict(self.x_test_noisy)
            decoded_noisy = self.autoencoder.decoder.predict(encoded_noisy)

        print("Mean activations: {}".format(encoded_imgs.mean()))
        
        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        def find_first_digit(d):
            return next(i for i in range(len(self.y_test)) if self.y_test[i] == d)

        indices = [find_first_digit(d) for d in range(10)]
        n = len(indices)  # how many digits we will display
        fig = plt.figure(figsize=(20, 4))

        rows = 3
        side = int(np.sqrt(self.autoencoder.encoding_dim))

        for i in range(n):
            
            if self.autoencoder.denoising:
                noisy_train = noise_input(self.x_train)
                ax = plt.subplot(rows, n, i + 1)
                plt.imshow(self.x_test_noisy[indices[i]].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
                # display encoding
                ax = plt.subplot(rows, n, i + 1 + n)
                plt.imshow(((encoded_noisy[indices[i]] + 1) / 2).reshape(side, side))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
                ax = plt.subplot(rows, n, i + 1 + 2 * n)
                plt.imshow(decoded_noisy[indices[i]].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            else:
                # display original
                ax = plt.subplot(rows, n, i + 1)
                plt.imshow(self.x_test[indices[i]].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            
                # display encoding
                ax = plt.subplot(rows, n, i + 1 + n)
                plt.imshow(((encoded_imgs[indices[i]] + 1) / 2).reshape(side, side))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(rows, n, i + 1 + 2 * n)
                plt.imshow(decoded_imgs[indices[i]].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                
        fig.savefig("/home/fdavidcl/Documentos/research/publications/2017/ReviewAutoencoders/examples/{}-{}.pdf".format(self.autoencoder.name, self.name), pad_inches = 0)

        return self

encoding_dim = 36
ae = Autoencoder(
    encoding_dim = encoding_dim,
    weight_decay = False
)

wd = Autoencoder(
    encoding_dim = encoding_dim,
    weight_decay = True
)

sae = Autoencoder(
    encoding_dim = encoding_dim,
    weight_decay = False,
    sparse = True,
    contractive = False,
    denoising = False,
    robust = False
)

cae = Autoencoder(
    encoding_dim = encoding_dim,
    weight_decay = False,
    sparse = False,
    contractive = True,
    denoising = False,
    robust = False
)

dae = Autoencoder(
    encoding_dim = encoding_dim,
    weight_decay = False,
    sparse = True,
    contractive = False,
    denoising = True,
    robust = False
)

rae = Autoencoder(
    encoding_dim = encoding_dim,
    weight_decay = True,
    sparse = False,
    contractive = False,
    denoising = False,
    robust = True
)

# for typ in [ae, wd, sae, cae, dae, rae]:
#     Trainer(typ).train(epochs = 60, optimizer = "adam").predict_test()

# for opt in ["sgd", "adam", "adagrad", "rmsprop", "adadelta"]:
#     Trainer(Autoencoder(
#         encoding_dim = encoding_dim,
#         weight_decay = False
#     )).train(epochs = 60, optimizer = opt).predict_test()

#for enc in [4, 16, 36, 81, 144]:
Trainer(Autoencoder(
    encoding_dim = 36,
    weight_decay = False,
    activation = "linear"
)).train(epochs = 60, loss = losses.mean_squared_error).predict_test()

# for act in ["relu", "sigmoid", "tanh", "selu"]:
#     Trainer(Autoencoder(
#         encoding_dim = encoding_dim,
#         weight_decay = False,
#         activation = act
#     )).train(epochs = 60).predict_test()
    
