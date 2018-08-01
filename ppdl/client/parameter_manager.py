from abc import ABCMeta, abstractmethod


class ParameterManager(metaclass=ABCMeta):

    @abstractmethod
    def upload_deltas(self, deltas, client_id):
        pass

    @abstractmethod
    def download_params(self):
        pass
