from abc import ABCMeta, abstractmethod


class GenericDataPreprocessor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self): raise NotImplementedError

    @abstractmethod
    def fit_transform(self, data): raise NotImplementedError

    @abstractmethod
    def transform(self, data): raise NotImplementedError


class ToNumpy(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, data):
        return data.values

    def transform(self, data):
        return data.values
