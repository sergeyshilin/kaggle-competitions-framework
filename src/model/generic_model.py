from abc import ABCMeta, abstractmethod


class GenericModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self): raise NotImplementedError

    @abstractmethod
    def fit(self, train, cv): raise NotImplementedError
