import os
import sys
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, path, *args, **kwargs):
        self.generator = None
        self.parameters = kwargs

        self.data_path = path
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.test_path = os.path.join(self.data_path, 'test.csv')

        # self._load_sample_data()
        self._load_data()

    def _load_data(self):
        sys.stdout.write("Loading training data... ")
        sys.stdout.flush()
        self.train = pd.read_csv(self.train_path)
        sys.stdout.write("Done\n")

        sys.stdout.write("Loading test data... ")
        sys.stdout.flush()
        self.test = pd.read_csv(self.test_path)
        sys.stdout.write("Done\n")

        self._process_target_column()
        self._process_id_column()

    def _apply_preprocess(self, method):
        processor = method()
        self.train = processor.fit_transform(self.train, self.train_y)
        self.test = processor.transform(self.test)

    def _process_target_column(self):
        if 'target' in self.parameters:
            self.target_name = self.parameters['target']
            self.train_y = self.train[self.target_name].values
            self.train.drop(self.target_name, axis=1, inplace=True)

    def _process_id_column(self):
        if 'target' in self.parameters:
            self.id_name = self.parameters['id']
            self.train_idx = self.train[self.id_name].values
            self.test_idx = self.test[self.id_name].values

    def freeze_data(self):
        self._frozen = (self.train, self.train_y, self.test,
            self.train_idx, self.test_idx)

    def generate_split(self, method, *args, **kwargs):
        self.generator = method(*args, **kwargs)

    def get_by_id(self, dtype, ids):
        if dtype == 'train' and isinstance(self.train, np.ndarray):
            return np.take(self.train, ids, axis=0), \
                np.take(self.train_y, ids, axis=0)
        elif dtype == 'train' and isinstance(self.train, pd.DataFrame):
            return self.train.iloc[ids], np.take(self.train_y, ids, axis=0)
        elif dtype == 'test' and isinstance(self.test, np.ndarray):
            return np.take(self.test, ids, axis=0)
        elif dtype == 'test' and isinstance(self.test, pd.DataFrame):
            return self.test.iloc[ids]

    def get_columns(self):
        return self.train.columns

    def get_parameter(self, parameter):
        return self.parameters[parameter]

    def get_split(self):
        if self.generator == None:
            raise Exception("Data generator is not set")

        return self.generator.split(self.train, self.train_y)

    def get_n_splits(self):
        if self.generator == None:
            raise Exception("Data generator is not set")

        return self.generator.get_n_splits()

    def get_target(self):
        return self.train_y

    def get_train(self):
        return self.train, self.train_y, self.train_idx

    def get_test(self):
        return self.test, self.test_idx

    def get_train_ids(self):
        return self.train_idx

    def get_test_ids(self):
        return self.test_idx

    def preprocess(self, *args):
        for method in args:
            self._apply_preprocess(method)

    def restore_frozen(self):
        self.train, self.train_y, self.test, self.train_idx, self.test_idx = \
            self._frozen
