from ppdl.grpc_server_client.client.client import Client
from ppdl.client.parameter_manager import ParameterManager
import torch
import time
import math
import os


class GrpcParameterManager(ParameterManager):

    def __init__(self, client_id=None, url=None):
        client_id = client_id or ("client_" + str(time.time()))
        self.grpc_client = Client(client_id=client_id, url=url)
        self.start_time = time.time()
        self.cycle_id, self.ttl, _ = self.grpc_client.download()

    def register_client(self, client_id):
        pass

    def upload_deltas(self, client_id, deltas):
        # TODO for now we just adjust the cycle ID if the cycle is already over
        # also we sleep until the next cycle if the TTL is dangerously low since the upload takes a few seconds
        if self.ttl < 5:
            time.sleep(self.ttl)
            self.ttl = 0
        time_diff = time.time() - self.start_time
        cycle_id = self.cycle_id 
        if time_diff >= self.ttl:
            cycle_id += math.ceil((time_diff - self.ttl) / int(os.environ["CYCLE_TIME"]))
        print("self.cycle_id = {}, time diff = {}, self.ttl = {}, new cycle_id = {}".format(self.cycle_id, time_diff, self.ttl, cycle_id))
        self.grpc_client.upload(cycle_id, {idx: val.item() for idx, val in enumerate(deltas)})

    def download_params(self, client_id):
        self.start_time = time.time()
        self.cycle_id, self.ttl, params = self.grpc_client.download()
        indices = torch.LongTensor(range(len(params)))
        weights = torch.zeros(len(params))
        i = 0
        for idx, val in params.items():
            indices[i] = idx
            weights[i] = val
        return (indices, weights)

