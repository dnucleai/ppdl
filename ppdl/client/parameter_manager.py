from abc import ABCMeta, abstractmethod


class ParameterManager(metaclass=ABCMeta):

    @abstractmethod
    def register_client(self, client_id):
        pass

    @abstractmethod
    def upload_deltas(self, client_id, deltas):
        pass

    @abstractmethod
    def download_params(self, client_id):
        pass
