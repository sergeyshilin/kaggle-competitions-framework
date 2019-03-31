import os
import sys
import numpy as np
import pandas as pd
import datetime
import shutil

from .generic_model import GenericModel


class ModelLoader:
    def __init__(self, model, model_params, *args, **kwargs):
        self.parameters = model_params
        self.model = model(*args, **kwargs)
        self._set_parameters()
        self.online_preprocessors = {}

    def _init_default_parameters(self):
        self.model_name = 'defaultmodel'
        self.fit_name = 'fit'
        self.predict_name = 'predict'
        self.preds_col_num = 0
        self.online_val_func = None

    def _apply_preprocessors(self, X, y, X_cv):
        for method, params in self.online_preprocessors.items():
            processor = method(**params)
            X, y = processor.fit_transform(X, y)
            X_cv = processor.transform(X_cv)

        return X, y, X_cv

    def _save_files(self, train, test, accuracy,
          caller_path, preds_path, models_path):
        model_name = self.get_parameter('name')
        acc_str = str(float('%.4f' % accuracy)).split('.')[-1]
        now = datetime.datetime.now()
        dt_str = now.strftime("%d%m%H%M%S")

        file_name = '_'.join([acc_str, model_name, dt_str])
        test_path = os.path.join(preds_path, 'test', file_name) + '.csv'
        train_path = os.path.join(preds_path, 'train', file_name) + '.csv'
        model_path = os.path.join(models_path, file_name) + '.py'

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        shutil.copyfile(caller_path, model_path)

    def _set_parameters(self):
        self._init_default_parameters()

        if 'name' in self.parameters:
            self.model_name = self.parameters['name']
        if 'fit' in self.parameters:
            self.fit_name = self.parameters['fit']
        if 'predict' in self.parameters:
            self.predict_name = self.parameters['predict']
        if 'pred_col' in self.parameters:
            self.preds_col_num = self.parameters['pred_col']
        if 'online_val' in self.parameters:
            self.online_val_func = self.parameters['online_val']

    def fit(self, train, cv, **kwargs):
        if not hasattr(self.model, self.fit_name):
            raise NotImplementedError("Fit method is not implemented")

        if isinstance(self.model, GenericModel):
            return getattr(self.model, self.fit_name)(train, cv)
        else:
            return getattr(self.model, self.fit_name)(*train, **kwargs)

    def get_parameter(self, parameter):
        return self.parameters[parameter]

    def get_preds_col_number(self):
        return self.preds_col_num

    def predict(self, *args, **kwargs):
        predict_result = getattr(self.model, self.predict_name)(*args, **kwargs)

        if len(predict_result.shape) > 1:
            return predict_result[:, self.get_preds_col_number()]

        return predict_result

    def preprocess_online(self, *args, **preprocessor_params):
        for method in args:
            self.online_preprocessors[method] = preprocessor_params

    def run(self, data_loader, evaluator, fit_params, predict_params,
          verbose=False):
        train_preds = np.zeros((len(data_loader.get_train_ids()), 1))
        test_preds = np.zeros((len(data_loader.get_test_ids()), 1))
        data_generator = data_loader.get_split()

        for fold, (tr_ind, cv_ind) in enumerate(data_generator):
            X_tr, y_tr = data_loader.get_by_id('train', tr_ind)
            X_cv, y_cv = data_loader.get_by_id('train', cv_ind)

            X_tr, y_tr, X_cv = self._apply_preprocessors(X_tr, y_tr, X_cv)

            if verbose: print("Start training the model '{}'... \n".format(
                self.model_name))

            if self.online_val_func:
                fit_params[self.online_val_func] = [(X_cv, y_cv)]

            self.fit((X_tr, y_tr), (X_cv, y_cv), **fit_params)

            preds_cv = self.predict(X_cv, **predict_params)
            preds_test_cur = self.predict(data_loader.test, **predict_params)

            accuracy = evaluator(y_cv, preds_cv)
            if verbose: print("Fold #{}: {}\n".format(fold + 1, accuracy))

            train_preds[cv_ind, :] = preds_cv.reshape((-1, 1))
            test_preds += preds_test_cur.reshape((-1, 1))

        test_preds = test_preds / data_loader.get_n_splits()

        overall = evaluator(data_loader.get_target(), train_preds)
        if verbose: print("Overall accuracy: {}\n".format(overall))
        return test_preds, train_preds, overall

    def save(self, data_loader, results, caller_path, preds_path, models_path):
        tr_idx = data_loader.get_train_ids()
        test_idx = data_loader.get_test_ids()
        id_col_name = data_loader.get_parameter('id')
        target_col_name = data_loader.get_parameter('target')
        preds_te, preds_tr, accuracy = results

        train_data = np.hstack(
            (
                tr_idx.reshape((-1, 1)),
                preds_tr.reshape((-1, 1))
            ))

        test_data = np.hstack(
            (
                test_idx.reshape((-1, 1)),
                preds_te.reshape((-1, 1))
            ))

        train_submit = pd.DataFrame(train_data,
            columns=[id_col_name, target_col_name])

        test_submit = pd.DataFrame(test_data,
            columns=[id_col_name, target_col_name])

        self._save_files(train_submit, test_submit, accuracy,
            caller_path, preds_path, models_path)
