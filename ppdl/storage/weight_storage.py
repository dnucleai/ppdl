from abc import ABCMeta, abstractmethod


class WeightStorage(metaclass=ABCMeta):

    @abstractmethod
    def store_weights(self, weights):
        pass

    @abstractmethod
    def fetch_weights(self):
        pass

    @abstractmethod
    def close(self):
        pass
