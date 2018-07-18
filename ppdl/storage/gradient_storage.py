from abc import ABCMeta, abstractmethod


class GradientStorage(metaclass=ABCMeta):

    @abstractmethod
    def store_gradients(self, indices, gradients):
        pass

    @abstractmethod
    def fetch_gradients(self, indices):
        pass