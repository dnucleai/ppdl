import os
import pickle
import numpy as np

from ppdl.storage.gradient_storage import GradientStorage


class LocalGradientStorage(GradientStorage):

    def __init__(self, filename):
        self.filename = filename

    # TODO: This should be moved to server side so clients can't reinitialize gradients
    def initialize_weights(self, shape):
        gradients = np.zeros(shape)
        with open(self.filename, 'wb') as f:
            pickle.dump(gradients)

    def update_weights(self, deltas):
        with open(self.filename, 'wb') as f:
            all_weights = self.fetch_all_weights()
            new_weights = []
            for weight, delta in zip(all_weights, deltas):
                new_weights.append(weight + delta)
            pickle.dump(new_weights)

    def fetch_all_weights(self):
        with open(self.filename, 'r') as f:
            return pickle.load(f)

    def fetch_weights(self, indices):
        with open(self.filename, 'r') as f:
            all_gradients = pickle.load(f)
            return all_gradients[indices]

    def cleanup(self):
        os.remove(self.filename)
