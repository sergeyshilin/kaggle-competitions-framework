from abc import ABCMeta, abstractmethod


class GenericDataPreprocessor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self): raise NotImplementedError

    @abstractmethod
    def fit_transform(self, X, y=None): raise NotImplementedError

    @abstractmethod
    def transform(self, X): raise NotImplementedError


class ToNumpy(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return X.values

    def transform(self, X):
        return X.values
