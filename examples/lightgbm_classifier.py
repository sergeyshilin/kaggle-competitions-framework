import os
import argparse
import shutil
import numpy as np
import pandas as pd

from data import DataLoader
from data.preprocessors import GenericDataPreprocessor, ToNumpy
from model import ModelLoader, GenericModel

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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from scipy.linalg import norm

class DropColumns(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X.drop(['ID_code'], axis=1)

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, ToNumpy)
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "lightgbm",
    'fit':           "fit",
    'predict':       "predict"
}

class LightGbmTrainer(GenericModel):
    def __init__(self):
        self.lgb_params = {
            "device": "gpu",
            "max_bin" : 63,
            "gpu_use_dp" : False,
            "objective" : "binary",
            "metric" : "auc",
            "boosting": 'gbdt',
            "max_depth" : 4,
            "num_leaves" : 13,
            "learning_rate" : 0.01,
            "bagging_freq": 10,
            "bagging_fraction" : 0.8,
            "feature_fraction" : 0.95,
            "min_data_in_leaf": 80,
            "tree_learner": "serial",
            "lambda_l1" : 5,
            "lambda_l2" : 5,
            "bagging_seed" : 42,
            "verbosity" : 0,
            "seed": 42
        }

    def fit(self, train, cv):
        x_tr, y_tr = train
        x_cv, y_cv = cv
        trn_data = lgb.Dataset(x_tr, label=y_tr)
        val_data = lgb.Dataset(x_cv, label=y_cv)
        evals_result = {}
        self.model = lgb.train(self.lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result)

    def predict(self, test):
        return self.model.predict(test)


model = ModelLoader(LightGbmTrainer, model_params)
results = model.run(data_loader, roc_auc_score, {}, {}, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
