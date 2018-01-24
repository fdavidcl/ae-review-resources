#!/usr/bin/env python

import numpy as np
np.random.seed(12345678) # for reproducibility

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import os
os.environ['PYTHONHASHSEED'] = '0'

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import regularizers, losses, callbacks, metrics
from keras import backend as K
from math import sqrt

import tensorflow as tf

from utils import *
import metrics

class MultilabelClassifier():
    def __init__(self, input_length, encoding_length, eval_encoding = True):
        self.build(input_length, encoding_length, eval_encoding)

    def build(self, input_length, encoding_length, eval_encoding):
        input_attributes = Input(shape = (input_length, ))
        #input_labels = Input(shape = (encoding_length, ))
        encoding = Dense(encoding_length
                         , activation = 'sigmoid'
#                         , activity_regularizer = activity_reg
#                         , kernel_regularizer = kernel_reg
                         , name = "encoding")(input_attributes)
        classification = Lambda(lambda x: K.round(x)
                                , name = "classification")(encoding)
        decodification = Dense(input_length
                               , activation = 'sigmoid'
                               , name = "decodification")(encoding)

        self.model = Model(input_attributes,
                           encoding if eval_encoding else classification)

        self.classifier = Model(input_attributes, classification)
        self.autoencoder = Model(input_attributes, decodification)

        return self

    def train(self, x_train, y_train,
              optimizer = 'adam',
              reconstruction_loss = losses.binary_crossentropy,
              classification_loss = losses.binary_crossentropy,
              weights = [1, 1],
              epochs = 60,
              pretrain = False):
        if pretrain:
            self.autoencoder.compile(optimizer = optimizer
                                     , loss = reconstruction_loss)
            self.autoencoder.fit(x_train, x_train,
                                 epochs = epochs,
                                 batch_size = 128,
                                 shuffle = True)
            
        self.model.compile(optimizer = optimizer
                           , loss = classification_loss)

        history = LossHistory()
        self.model.fit(x_train, y_train,
                       epochs = epochs,
                       batch_size = 256,
                       shuffle = True,
                       callbacks = [history])
