from abc import ABCMeta, abstractmethod


class GradientStorage(metaclass=ABCMeta):

    # TODO: This should be moved to server side so clients can't reinitialize gradients
    @abstractmethod
    def initialize_gradients(self, shape):
        pass

    @abstractmethod
    def store_gradients(self, indices, gradients):
        pass

    @abstractmethod
    def fetch_gradients(self, indices):
        pass
