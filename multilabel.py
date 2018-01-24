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

def microprecision(y_true, y_pred):
    tp = K.sum(y_true * y_pred)
    fp = K.sum(K.clip(y_pred - y_true, 0, 1))
    return -tp / (tp + fp)

def microrecall(y_true, y_pred):
    tp = K.sum(y_true * y_pred)
    fn = K.sum(K.clip(y_true - y_pred, 0, 1))
    return -tp / (tp + fn)

def microfm(y_true, y_pred):
    prec = microprecision(y_true, y_pred)
    recl = microrecall(y_true, y_pred)
    return 2 * prec * recl / (prec + recl)

class MultilabelAutoencoder():
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
                           [decodification, encoding if eval_encoding else classification])

        self.classifier = Model(input_attributes, classification)
        self.autoencoder = Model(input_attributes, decodification)

        return self

    def train(self, x_train, y_train,
              optimizer = 'adam',
              reconstruction_loss = losses.binary_crossentropy,
              classification_loss = losses.binary_crossentropy,
              weights = [1, 1],
              epochs = 60):
        self.model.compile(optimizer = optimizer
                           , loss = [reconstruction_loss, classification_loss]
                           , loss_weights = weights)

        history = LossHistory()
        self.model.fit(x_train,
                       [x_train, y_train],
                       epochs = epochs,
                       batch_size = 256,
                       shuffle = True,
                       callbacks = [history])

from scipy.io import arff
from xml.etree import ElementTree
# TODO: implement labels at front
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

from weka.arff import ArffFile, Nominal, String, Numeric
# TODO: implement labels at front
def load_sparse_arff(filename, labelcount, labels_at_end = True):
    fil = ArffFile.load(filename)
    attr_names = fil.attributes
    input_len = len(attr_names) - labelcount
    attr_to_index = {}

    for idx, n in enumerate(attr_names):
        attr_to_index[n] = idx

    x = np.zeros((len(fil.data), input_len), dtype = np.float32)
    y = np.zeros((len(fil.data), labelcount), dtype = np.int8)

    nom_to_num = {'0': 0, '1': 1}

    for instance in fil.data:
        for attr, val in instance.items():
            idx = attr_to_index[attr]
        
            if idx > input_len:
                idx = idx - input_len
                y[idx] = nom_to_num[val.value]                
            else:
                x[idx] = val.value if type(val) == Numeric else nom_to_num[val.value]
                
    return x, y

from sklearn.preprocessing import normalize


FILE = "partitions/CAL500-5x2x1-1"
SPARSE = False
LABELS = 174

load_f = load_sparse_arff if SPARSE else load_arff
    
med_x, med_y = load_f(FILE + "tra.arff", LABELS)
test_x, test_y = load_f(FILE + "tst.arff", LABELS)

med_x = normalize(med_x, axis = 0)
test_x = normalize(test_x, axis = 0)

for w in [1]:
    mlae = MultilabelAutoencoder(med_x.shape[1], med_y.shape[1], eval_encoding = False)
    mlae.train(
        med_x, med_y,
        weights = [1, w],
        epochs = 1000,
        classification_loss = losses.binary_crossentropy
    )
    
    pred_y = mlae.classifier.predict(med_x)
    #pred_y = np.int64(pred_y)
    print("TRAIN {} ============".format(w))
    # print(pred_y.shape)
    #print(losses.binary_crossentropy(med_y, pred_y))
    metrics.report(med_y, pred_y)

    pred_y = mlae.classifier.predict(test_x)
    #pred_y = np.int64(pred_y)
    print("TEST {} =============".format(w))
    # print(pred_y.shape)
    metrics.report(test_y, pred_y)
