import torch
import torch.nn as nn
import function_mapper as fm

class CNNModel (nn.Module):
    def __init__(self, model):
        super(CNNModel, self).__init__()
        self.functions = model.function


    def forward(self, x):
        for function in self.functions:
            x = fm.apply_torch_function(function, self.training, x)
        return x

    def is_nn(self):
        return True

    def train(self, args, device, train_loader, optimizer, epoch):
        return
