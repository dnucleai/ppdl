import torch


def flatten_parameters(params):
    return torch.cat([p.data.view(-1) for p in params])