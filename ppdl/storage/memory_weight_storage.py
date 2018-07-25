from ppdl.storage.weight_storage import WeightStorage


class MemoryWeightStorage(WeightStorage):

    def __init__(self):
        self.weights = None
        self.stats = None

    def store_weights(self, weights):
        self.weights = weights

    def store_stats(self, stats):
        self.stats = stats

    def fetch_weights(self):
        return self.weights

    def fetch_stats(self):
        return self.stats

    def close(self):
        pass
