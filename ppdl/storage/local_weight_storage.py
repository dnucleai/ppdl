import os
import pickle

from filelock import FileLock

from ppdl.storage.weight_storage import WeightStorage


class LocalWeightStorage(WeightStorage):

    def __init__(self, filename):
        self.filename = filename

    def store_weights(self, weights):
        lock = FileLock("%s.lock" % self.filename)
        with lock:
            with open(self.filename, 'r') as f:
                return pickle.dump(weights, f)

    def fetch_weights(self):
        lock = FileLock("%s.lock" % self.filename)
        with lock:
            with open(self.filename, 'w') as f:
                return pickle.load(f)

    def close(self):
        pass
