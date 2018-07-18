import os
import pickle
import numpy as np

from ppdl.storage.gradient_storage import GradientStorage


class LocalGradientStorage(GradientStorage):

    def __init__(self, filename):
        self.filename = filename

    # TODO: This should be moved to server side so clients can't reinitialize gradients
    def initialize_gradients(self, shape):
        gradients = np.zeros(shape)
        with open(self.filename, 'wb') as f:
            pickle.dump(gradients)

    def store_gradients(self, indices, gradients):
        with open(self.filename, 'wb') as f:
            all_gradients = self.fetch_all_gradients()
            all_gradients[indices] = gradients
            pickle.dump(all_gradients)

    def fetch_all_gradients(self):
        with open(self.filename, 'r') as f:
            return pickle.load(f)

    def fetch_gradients(self, indices):
        with open(self.filename, 'r') as f:
            all_gradients = pickle.load(f)
            return all_gradients[indices]

    def cleanup(self):
        os.remove(self.filename)
