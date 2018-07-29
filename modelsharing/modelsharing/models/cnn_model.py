import torch
import torch.nn as nn
import function_service as fs

class CNNModel (nn.Module):
    def __init__(self, model):
        super(CNNModel, self).__init__()
        self.functions = model.function

    def forward(self, x):
        for function in self.functions:
            x = fs.apply_torch_function(function, x)
        return x

    def is_nn(self):
        return True

    def train(self, args, device, train_loader, optimizer, epoch):
        return
