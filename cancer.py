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
from scipy.io import arff

from utils import *
from autoencoder import Autoencoder

from keras.datasets import mnist

def load_cancer():
    with open("wdbc.arff") as arff_file:
        data, meta = arff.loadarff(arff_file)

    classes = { b'M': 1, b'B': 0 }
    y = [classes[i[0]] for i in data]
    data = data.tolist()
    x = np.asarray([list(i[1:31]) for i in data])
    
    return x, y
    

from sklearn.preprocessing import normalize

class CancerTrainer():
    def __init__(self, autoencoder = Autoencoder()):
        self.autoencoder = autoencoder
        #(x_train, _), (x_test, self.y_test) = mnist.load_data()

        x_train, self.y_test = load_cancer()
        
        self.x_train = normalize(x_train, axis = 0)
        self.x_test = self.x_train

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
                                           callbacks=[history])
        else:
            self.autoencoder.model.fit(self.x_train, self.x_train,
                                       epochs = epochs,
                                       batch_size = 256,
                                       shuffle = True,
                                       callbacks=[history])

        self.name = "{}-{}".format(
            optimizer,
            "mse" if loss == losses.mean_squared_error else ("xent" if loss == losses.binary_crossentropy or self.autoencoder.contractive else "corr")
        )

        with open("cancer-{}-{}.csv".format(self.autoencoder.name, self.name), "w") as out_file:
            out_file.write(",".join(("{}".format(x) for x in history.losses)))

        return self

    def predict_test(self):
        # encode and decode some instances
        # note that we take them from the *test* set
        encoded_instances = self.autoencoder.encoder.predict(self.x_train)
        with open("cancer-encoded-{}-{}.csv".format(self.autoencoder.name, self.name), "w") as out_file:
            out_file.write(
                "\n".join(
                    ",".join(
                        ("{}".format(x) for x in instance)
                    )
                    for instance in encoded_instances
                )
            )
        print("Mean activations: {}".format(encoded_instances.mean()))
        
        return self

CancerTrainer(Autoencoder(
    input_dim = 30,
    encoding_dim = 2,
    weight_decay = True,
    sparse = False,
    activation = "linear"
)).train(epochs = 300, loss = losses.mean_squared_error).predict_test()

