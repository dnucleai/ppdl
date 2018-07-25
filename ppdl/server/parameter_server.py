import torch

class ParameterServer:

    def __init__(self, weight_storage):
        self.weight_storage = weight_storage

    def initialize_weights(self, size):
        weights = torch.zeros(size)
        stats = torch.zeros(size)
        self.weight_storage.store_weights(weights)
        self.weight_storage.store_stats(stats)

    def update_weights(self, deltas):
        all_weights = self.fetch_weights()
        new_weights = all_weights + deltas
        self.weight_storage.store_weights(new_weights)

    def fetch_weights(self):
        return self.weight_storage.fetch_weights()
