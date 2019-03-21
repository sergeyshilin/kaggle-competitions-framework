import os
import argparse
import shutil
import numpy as np
import pandas as pd

from data import DataLoader
from data.preprocessors import GenericDataPreprocessor, ToNumpy
from model import ModelLoader

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--data', required=True,
    help='Path where to read train.csv and test.csv files')
parser.add_argument('--models', required=True,
    help='Path where to save source code')
parser.add_argument('--preds', required=True,
    help='Path where to save model outputs')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)
args = parser.parse_args()


## >> Read and preprocess data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

class DropColumns(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, data):
        return data.drop(['ID_code'], axis=1)

    def transform(self, data):
        return data.drop(['ID_code'], axis=1)

class FeatureEngineering(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, data):
        for c in data.columns:
            data['squared_' + c] = data[c] ** 2
            data['cubic_' + c] = data[c] ** 3
            data['round2_' + c] = np.round(data[c], 2)
        return data

    def transform(self, data):
        return self.fit_transform(data)

class ReshapeToNnInput(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, data):
        return data.reshape((-1, data.shape[-1], 1))

    def transform(self, data):
        return data.reshape((-1, data.shape[-1], 1))

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns)
data_loader.preprocess(FeatureEngineering)
data_loader.preprocess(ToNumpy, StandardScaler)
# data_loader.preprocess(ReshapeToNnInput)
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.losses import binary_crossentropy
from keras import regularizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier


def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def dense_nn_model():
    layer_size = [16, 8, 8, 4]
    model = Sequential()

    for i, nodes in enumerate(layer_size):
        model.add(Dense(
            nodes,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(0.01),
            name='dense_{}'.format(i)))
        model.add(Dropout(rate=0.1))

    model.add(Dense(1, kernel_initializer='glorot_uniform',
        activation='sigmoid', name='output'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=[keras_auc, 'accuracy'])
    return model


model_params = {
    'name':     "dense_nn",
    'fit':      "fit",
    'predict':  "predict_proba",
    'pred_col': 1
}

nn_params = {
    'build_fn': dense_nn_model,
    'epochs': 25,
    'batch_size': 256,
    'verbose': 1
}

np.random.seed(42)

model = ModelLoader(KerasClassifier, model_params, **nn_params)

fit_params = {}; predict_params = {}
results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path,
        args.preds, args.models)
## << Create and train model
