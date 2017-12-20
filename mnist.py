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

from utils import *
from autoencoder import Autoencoder

from keras.datasets import mnist

class MNISTTrainer():
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

        with open("{}-{}.csv".format(self.autoencoder.name, self.name), "w") as out_file:
            out_file.write(",".join(("{}".format(x) for x in history.losses)))

        return self

    def predict_test(self):
        # encode and decode some digits
        # note that we take them from the *test* set
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

                
        fig.savefig("{}-{}.pdf".format(self.autoencoder.name, self.name), pad_inches = 0)

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

for typ in [ae, wd, sae, cae, dae, rae]:
    MNISTTrainer(typ).train(
        epochs = 60, optimizer = "rmsprop"
    ).predict_test()

