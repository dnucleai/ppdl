import pickle

from filelock import FileLock

from ppdl.storage.weight_storage import WeightStorage


class LocalWeightStorage(WeightStorage):

    def __init__(self, weights_filename, stats_filename):
        self.weights_filename = weights_filename
        self.stats_filename = stats_filename

    def store_weights(self, weights):
        lock = FileLock("%s.lock" % self.weights_filename)
        with lock:
            with open(self.weights_filename, 'r') as f:
                return pickle.dump(weights, f)

    def store_stats(self, stats):
        lock = FileLock("%s.lock" % self.stats_filename)
        with lock:
            with open(self.stats_filename, 'r') as f:
                return pickle.dump(stats, f)

    def fetch_weights(self):
        lock = FileLock("%s.lock" % self.weights_filename)
        with lock:
            with open(self.weights_filename, 'w') as f:
                server_params = pickle.load(f)
                return server_params['weights']

    def fetch_stats(self):
        lock = FileLock("%s.lock" % self.stats_filename)
        with lock:
            with open(self.stats_filename, 'w') as f:
                server_params = pickle.load(f)
                return server_params['stats']

    def close(self):
        pass
