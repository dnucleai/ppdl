import torch


class ParameterServer:

    def __init__(self, weight_storage, num_clients):
        self.weight_storage = weight_storage
        self.num_clients = num_clients
        self.client_state = {}

    def initialize_weights(self, size):
        weights = torch.zeros(size)
        stats = torch.zeros(size)
        self.weight_storage.store_weights(weights)
        self.weight_storage.store_stats(stats)

    def register_client(self, client_id):
        self.client_state[client_id] = 1

    def can_download(self):
        if len(self.client_state) != self.num_clients:
            return False
        temp_state = None
        for state in self.client_state.values():
            if temp_state is None:
                temp_state = state
            elif state != temp_state:
                return False
        return True

    def update_weights(self, deltas, client_id):
        all_weights = self.weight_storage.fetch_weights()
        new_weights = all_weights + deltas
        self.weight_storage.store_weights(new_weights)
        self.client_state[client_id] = True

    def fetch_weights(self, client_id):
        self.client_state[client_id] = False
        return self.weight_storage.fetch_weights()
