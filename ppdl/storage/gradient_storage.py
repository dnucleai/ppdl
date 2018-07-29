from abc import ABCMeta, abstractmethod


class GradientStorage(metaclass=ABCMeta):

    # TODO: This should be moved to server side so clients can't reinitialize gradients
    @abstractmethod
    def initialize_weights(self, shape):
        pass

    @abstractmethod
    def update_weights(self, deltas):
        pass

    @abstractmethod
    def fetch_weights(self, indices):
        pass
