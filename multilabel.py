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
#                               , activation = 'sigmoid'
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

    for row, instance in enumerate(fil.data):
        for attr, val in instance.items():
            idx = attr_to_index[attr]

            if labels_at_end:
                if idx > input_len:
                    idx = idx - input_len
                    y[row, idx] = nom_to_num[val.value]                
                else:
                    x[row, idx] = val.value if type(val) == Numeric else nom_to_num[val.value]
            else:
                if idx <= labelcount:
                    y[row, idx] = nom_to_num[val.value]
                else:
                    idx = idx - labelcount
                    x[row, idx] = val.value if type(val) == Numeric else nom_to_num[val.value]
                    
    return x, y

import arff
def load_liac_arff(filename, labelcount, labels_at_end = True):
    ardata = arff.load(open(filename, "r"))
    input_len = len(ardata["attributes"]) - labelcount
    instances = len(ardata["data"])
    
    x = np.zeros((instances, input_len), dtype = np.float32)
    y = np.zeros((instances, labelcount), dtype = np.int8)

    nom_to_num = {'0': 0, '1': 1, 'YES': 0, 'NO': 1}

    for row, instance in enumerate(ardata["data"]):
        if labels_at_end:
            for idx, val in enumerate(instance):
                if idx < input_len:
                    x[row, idx] = nom_to_num[val] if type(val) == str else val
                else:
                    idx = idx - input_len
                    y[row, idx] = nom_to_num[val]
        else:
            for idx, val in enumerate(instance):
                if idx >= labelcount:
                    idx = idx - labelcount
                    x[row, idx] = nom_to_num[val] if type(val) == str else val
                else:
                    y[row, idx] = nom_to_num[val]

    return x, y

from sklearn.preprocessing import normalize


datasets = {
    "bibtex": (22, True),
    "medical": (45, True),
    "emotions": (6, True),
    "CAL500": (174, True),
    "corel5k": (374, True)
}

chosen = "corel5k"

#load_f = load_sparse_arff if SPARSE else load_arff
    
train_x, train_y = load_liac_arff("partitions/" + chosen + "-5x2x1-1tra.arff", datasets[chosen][0], labels_at_end = datasets[chosen][1])
test_x, test_y = load_liac_arff("partitions/" + chosen + "-5x2x1-1tst.arff", datasets[chosen][0], labels_at_end = datasets[chosen][1])

#print(train_x[0:10])

#exit(0)
#med_x = normalize(med_x, axis = 0)
#test_x = normalize(test_x, axis = 0)

for w in [10]:
    mlae = MultilabelAutoencoder(
        train_x.shape[1], train_y.shape[1],
        eval_encoding = True
    )
    mlae.train(
        train_x, train_y,
        weights = [1, w],
        epochs = 400000 // train_x.shape[0],
        classification_loss = losses.binary_crossentropy,
#        reconstruction_loss = losses.mean_squared_error
        reconstruction_loss = losses.binary_crossentropy
    )
    
    pred_y = mlae.classifier.predict(train_x)
    #pred_y = np.int64(pred_y)
    print("TRAIN {} ============".format(w))
    # print(pred_y.shape)
    #print(losses.binary_crossentropy(med_y, pred_y))
    metrics.report(train_y, pred_y)

    pred_y = mlae.classifier.predict(test_x)
    #pred_y = np.int64(pred_y)
    print("TEST {} =============".format(w))
    # print(pred_y.shape)
    metrics.report(test_y, pred_y)
