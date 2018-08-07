import torch


class ParameterServer:

    def __init__(self, weight_storage, num_clients, theta):
        self.weight_storage = weight_storage
        self.num_clients = num_clients
        self.client_state = {}
        self.theta = theta

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

    def update_weights(self, client_id, deltas):
        all_weights = self.weight_storage.fetch_weights()
        new_weights = all_weights + deltas

        updated_indices = torch.nonzero(deltas)
        stats = self.weight_storage.fetch_stats()
        stats[updated_indices] += 1
        self.weight_storage.store_stats(stats)

        self.weight_storage.store_weights(new_weights)
        self.client_state[client_id] = True

    def fetch_weights(self, client_id):
        self.client_state[client_id] = False
        weights = self.weight_storage.fetch_weights()
        stats = self.weight_storage.fetch_stats()

        threshold = round(self.theta * stats.size())
        _, indices = torch.topk(stats, threshold)
        indices = indices.view(-1)
        selected_weights = weights[indices]
        return {
            "indices": indices,
            "weights": selected_weights
        }
