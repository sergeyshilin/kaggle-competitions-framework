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
parser.add_argument('--validate', dest='predict', action='store_false')
parser.add_argument('--predict', dest='predict', action='store_true')
parser.set_defaults(predict=True)
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

dl_params = {'target': "target"}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, ToNumpy, StandardScaler)
data_loader.generate_split(StratifiedShuffleSplit,
    n_splits=5,
    test_size=0.2,
    random_state=42)
## << Read and preprocess data


def run_cross_validation(model, data_loader):
    data_generator = data_loader.get_split()
    cur_fold = 1
    auc_list = []

    for tr_ind, cv_ind in data_generator:
        X_tr, y_tr = data_loader.get_by_id('train', tr_ind)
        X_cv, y_cv = data_loader.get_by_id('train', cv_ind)

        model.fit(X_tr, y_tr)
        preds = model.predict(X_cv)
        preds = preds[:, model.get_preds_col_number()]

        fpr, tpr, _ = roc_curve(y_cv, preds)
        accuracy = auc(fpr, tpr)

        print("Fold #{}: {}".format(cur_fold, accuracy))
        auc_list.append(accuracy)
        cur_fold += 1

    overall = np.array(auc_list).mean()
    print("Overall AUC: {}\n".format(overall))


def run_train_predict(model, data_loader):
    train_x, train_y, tr_idx = data_loader.get_train()
    test = data_loader.get_test()

    model.fit(train_x, train_y)
    preds_test = model.predict(test)
    preds_test = preds_test[:, model.get_preds_col_number()]

    preds_train = model.predict(train_x)
    preds_train = preds_train[:, model.get_preds_col_number()]

    fpr, tpr, _ = roc_curve(train_y, preds_train)
    accuracy = auc(fpr, tpr)
    print("Training accuracy: {}\n".format(accuracy))

    return preds_test, preds_train, tr_idx, accuracy


def save_results(model, data_loader, results):
    import datetime

    preds_te, preds_tr, tr_idx, accuracy = results
    submission = data_loader.get_sample_submission()

    test_submit = submission.copy()
    target_col_name = data_loader.get_parameter('target')
    test_submit[target_col_name] = preds_te

    train_data = np.hstack(
        (
            tr_idx.reshape((-1, 1)),
            preds_tr.reshape((-1, 1))
        ))

    train_submit = pd.DataFrame(train_data, columns=["Id", "Preds"])

    model_name = model.get_parameter('name')
    acc_str = str(float('%.4f' % accuracy)).split('.')[-1]
    now = datetime.datetime.now()
    dt_str = now.strftime("%d%m%H%M%S")

    file_name = '_'.join([acc_str, model_name, dt_str])
    test_path = os.path.join(args.preds, 'test', file_name) + '.csv'
    train_path = os.path.join(args.preds, 'train', file_name) + '.csv'
    model_path = os.path.join(args.models, file_name) + '.py'

    test_submit.to_csv(test_path, index=False)
    train_submit.to_csv(train_path, index=False)
    shutil.copyfile(os.path.basename(__file__), model_path)


## >> Create and train model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

model_params = {
    'name':     "gaussian",
    'fit':      "fit",
    'predict':  "predict_proba",
    'pred_col': 1
}
model = ModelLoader(GaussianNB, model_params)

if not args.predict:
    run_cross_validation(model, data_loader)
else:
    results = run_train_predict(model, data_loader)
    save_results(model, data_loader, results)
## << Create and train model
