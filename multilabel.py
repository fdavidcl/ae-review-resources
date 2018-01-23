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
from autoencoder import Autoencoder


from keras.datasets import mnist

def hamming_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

class MultilabelAutoencoder():
    ## I need a model graph with two outputs (the encoding layer as well as the actual output), two loss functions (possibly cross entropy) and a pair of loss weights.
    def __init__(self, input_length, encoding_length):
        self.build(input_length, encoding_length)

    def build(self, input_length, encoding_length):
        input_attributes = Input(shape = (input_length, ))
        input_labels = Input(shape = (encoding_length, ))
        encoding = Dense(encoding_length
                         , activation = 'sigmoid'
#                         , activity_regularizer = activity_reg
#                         , kernel_regularizer = kernel_reg
                         , name = "encoding")(input_attributes)
        classification = Lambda(lambda x: tf.round(x)
                                , name = "classification")(encoding)
        decodification = Dense(input_length
                               , activation = 'sigmoid'
                               , name = "decodification")(encoding)

        self.model = Model([input_attributes, input_labels],
                           [decodification, classification])

        self.classifier = Model(input_attributes, classification)
        self.autoencoder = Model(input_attributes, decodification)

        return self

    def train(self, x_train, y_train,
              optimizer = 'rmsprop',
              loss = losses.binary_crossentropy,
              clas_weight = 0.5,
              epochs = 60):
        self.model.compile(optimizer = optimizer
                           , loss = [loss, hamming_loss]
                           , loss_weights = [1 - clas_weight, clas_weight])

        history = LossHistory()
        self.model.fit([x_train, y_train],
                       [x_train, y_train],
                       epochs = epochs,
                       batch_size = 256,
                       shuffle = True,
                       callbacks = [history])

from scipy.io import arff
from xml.etree import ElementTree
def load_arff(filename, labelcount, labels_at_end = True):
    with open(filename) as arff_file:
        data, meta = arff.loadarff(arff_file)
#    label_data = ElementTree.parse(xmlfile).getroot()
#    label_names = [label.get("name") for label in label_data.getchildren()]
    data = data.tolist()
    y_len = labelcount
    x_len = len(data[0]) - y_len
    x = np.asarray([list(i[0:x_len]) for i in data])
    classes = { b'1': 1, b'0': 0 }
    y = np.asarray([[classes[y] for y in i[x_len:(x_len + y_len)]] for i in data])
    return x, y

from sklearn.preprocessing import normalize

# mediamill
med_x, med_y = load_arff("mediamill-train.arff", 101)
test_x, test_y = load_arff("mediamill-test.arff", 101)
med_x = normalize(med_x, axis = 0)
test_x = normalize(test_x, axis = 0)
mlae = MultilabelAutoencoder(med_x.shape[1], med_y.shape[1])
mlae.train(med_x, med_y, clas_weight = 0, epochs = 40)

pred_y = mlae.classifier.predict(med_x)
#pred_y = np.int64(pred_y)
print("TRAIN ============")
print(pred_y.shape)
#print(losses.binary_crossentropy(med_y, pred_y))
metrics.report(med_y, pred_y)

pred_y = mlae.classifier.predict(test_x)
#pred_y = np.int64(pred_y)
print("TEST =============")
print(pred_y.shape)
#print(losses.binary_crossentropy(test_y, pred_y))
metrics.report(test_y, pred_y)
