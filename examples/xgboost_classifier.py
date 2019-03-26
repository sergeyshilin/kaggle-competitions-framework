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
import xgboost as xgb
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "xgboost",
    'fit':           "fit",
    'predict':       "predict_proba",
    'pred_col':      1,
    'online_val':    "eval_set"
}

xgboost_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True,
    'booster': 'gbtree',
    'n_jobs': -1,
    'n_estimators': 20000,
    'grow_policy': 'lossguide',
    'max_depth': 5,
    'max_delta_step': 2,
    'seed': 42,
    'colsample_bylevel': 0.7,
    'colsample_bytree': 0.1,
    'gamma': 1.5,
    'learning_rate': 0.02,
    'max_leaves': 23,
    'min_child_weight': 64,
    'reg_alpha': 0.95,
    'reg_lambda': 50.0,
    'subsample': 0.25
}

model = ModelLoader(xgb.XGBClassifier, model_params, **xgboost_params)

fit_params = {
    'early_stopping_rounds': 2500,
    'verbose': 1000
}
predict_params = {}

results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
