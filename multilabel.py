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
from traditional_mlp import MultilabelClassifier

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

        self.encoder = Model(input_attributes, encoding)
        self.classifier = Model(input_attributes, classification)
        self.autoencoder = Model(input_attributes, decodification)

        return self

    def train(self, x_train, y_train,
              optimizer = 'adam',
              reconstruction_loss = losses.binary_crossentropy,
              classification_loss = losses.binary_crossentropy,
              weights = [1, 1],
              epochs = 60,
              post_train = False):
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

        # Post-training worsens good results
        if post_train:
            self.encoder.compile(optimizer = optimizer,
                                 loss = classification_loss)
            self.encoder.fit(x_train, y_train, epochs = epochs,
                             batch_size = 256, shuffle = True)



import arff # needs liac-arff
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

# from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

datasets = {
    "bibtex": (22, True),
    "CAL500": (174, True),
    "corel5k": (374, True),
    "emotions": (6, True),
    "enron": (53, True),
    "mediamill": (101, True),
    "medical": (45, True),
    "scene": (6, True),
    "SLASHDOT-F": (22, False),
    "tmc2007": (22, True) # tmc2007_500 in Cometa
}

def experiment(model_c, dataset_name, train_path, test_path, clas_weight, scale = False, loss_mse = False, configuration = "", val_step = None):
    train_x, train_y = load_liac_arff(train_path, datasets[dataset_name][0], labels_at_end = datasets[dataset_name][1])
    test_x, test_y = load_liac_arff(test_path, datasets[dataset_name][0], labels_at_end = datasets[dataset_name][1])

    if scale:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    mlae = model_c(
        train_x.shape[1], train_y.shape[1],
        eval_encoding = True
    )
    epochs = 600000 // train_x.shape[0]
    mlae.train(
        train_x, train_y,
        weights = [1, clas_weight],
        epochs = epochs,
        classification_loss = losses.binary_crossentropy,
        reconstruction_loss = losses.mean_squared_error if loss_mse else losses.binary_crossentropy
    )
    
    print("TRAIN ============")
    pred_y = mlae.classifier.predict(train_x)
    # print(pred_y.shape)
    metrics.report(train_y, pred_y)
    
    print("TEST =============")
    pred_y = mlae.classifier.predict(test_x)
    # print(pred_y.shape)
    metrics.report(test_y, pred_y)
    
    metrics.csv_report("posttrained.csv", dataset_name + configuration + "_w{}_e{}".format(clas_weight, epochs), test_y, pred_y, val_step)


numerical = ["CAL500", "emotions", "mediamill", "scene"]
binary = ["bibtex", "corel5k", "enron", "medical", "SLASHDOT-F", "tmc2007"]

numerical_weights = [1, 10, 100]
binary_weights = [0.1, 0.5, 1, 2, 10]

def validation(dataset, scale = False, loss_mse = False):
    for val in [1, 2]:
        for step in range(1, 6):
            train_path = "partitions/{}-5x2x{}-{}tra.arff".format(dataset, val, step)
            test_path = "partitions/{}-5x2x{}-{}tst.arff".format(dataset, val, step)

            for net in [MultilabelAutoencoder, MultilabelClassifier]:
                configuration = "_mse" if loss_mse else "_xent"
                if net == MultilabelClassifier:
                    configuration = configuration + "_mlp"
                    my_weights = [1]
                else:
                    my_weights = numerical_weights if loss_mse else binary_weights
            
                for w in my_weights:
                    experiment(net, dataset, train_path, test_path, w, scale = scale, loss_mse = loss_mse, configuration = configuration, val_step = (5 if val == 2 else 0) + step)


#for dataset in binary:
#    validation(dataset, False, False)

validation("SLASHDOT-F", False, False)

#for dataset in numerical:
#    validation(dataset, False, True)
