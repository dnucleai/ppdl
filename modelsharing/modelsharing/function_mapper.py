import torch
import torch.nn as nn
import torch.nn.functional as F
import model_pb2 as mpb2

class FunctionMapper():
    def __init__(self):
        self.function_map = {
            'Conv1d': nn.Conv1d,
            'Conv2d': nn.Conv2d,
            'Conv3d': nn.Conv3d,
            'Dropout2d': nn.Dropout2d,
            'max_pool2d': F.max_pool2d,
            'relu': F.relu,
            'view': self.apply_view,
            'Linear': nn.Linear,
            'dropout': self.apply_dropout,
            'log_softmax': F.log_softmax,
            'l1_loss': F.l1_loss,
            'adaptive_avg_pool1d': F.adaptive_avg_pool1d,
            'adaptive_avg_pool2d': F.adaptive_avg_pool2d,
            'adaptive_avg_pool3d': F.adaptive_avg_pool3d,
            'avg_pool1d': F.avg_pool1d,
            'avg_pool2d': F.avg_pool2d,
            'avg_pool3d': F.avg_pool3d,
            'hardtanh': F.hardtanh,
            'leaky_relu': F.leaky_relu,
            'tanh': F.tanh,
            'normalize': F.normalize,
            'sigmoid': F.sigmoid,
            'num_flat_features': self.num_flat_features,
        }
        self.class_map = {
            'tensor': torch.Tensor,
            'Conv1d': nn.Conv1d,
            'Dropout2d': nn.Dropout2d,
            'Linear': nn.Linear
        }

        self.string_set = set([
            'training'
        ])

        self.input = None

    def set_input(self, input):
        self.input = input

    def get_function( self, function_string ):
        return self.function_map[function_string]

    def apply_function(self, func, args):
        try:
            return self.call_function( func, args )
        except (RuntimeError):
            raise ValueError("Function specified by message does not exist or has an incorrect number of arguments.")


    def call_function(self, func, args):
        if len(args) == 4:
            return func(args[0], args[1], args[2], args[3])
        elif len(args) == 3:
            return func(args[0], args[1], args[2])
        elif len(args) == 2:
            return func(args[0], args[1])
        elif len(args) == 1:
            if self.is_empty(args[0]):
                return func()
            return func(args[0])
        else:
            return func

    def convert_args(self, args):
        converted_args = []
        flags = []
        for arg in args:
            if self.represents_int(arg):
                converted_args.append(int(arg))
            elif self.in_function_map(arg):
                converted_args.append(self.function_map[arg]())
            elif self.represents_class(arg):
                converted_args.append(arg)
            elif self.is_empty(arg):
                converted_args.append(arg)
            elif self.in_string_set(arg):
                converted_args.append(arg)
                flags.append(arg)
            elif self.represents_bool(arg):
                converted_args.append(bool(arg))
            else:
                raise ValueError("Argument for called function is not properly specified, or is not yet implemented.")
        return converted_args, flags

    def represents_int(self, s):
        try:
            val = int(s)
            return True
        except (ValueError, TypeError):
            return False

    def represents_bool(self, s):
        try:
            val = bool(s)
            return True
        except (ValueError, TypeError):
            return False

    def represents_class(self, cl):
        for c in self.class_map.values():
            if isinstance(cl, c):
                return True
        return False

    def is_empty(self, arg):
        if arg is None:
            return True
        try:
            val = str(arg)
            if val == "":
                return True
            else:
                return False
        except (ValueError, TypeError):
            return False

    def in_function_map(self, input):
        return input in self.function_map.keys()

    def in_string_set(self, arg):
        return arg in self.string_set

    def num_flat_features(self):
        if self.input is None:
            return
        size = self.input.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def apply_dropout(self, *args):
        if len(args) != 4:
            raise ValueError("Insuffient args for dropout function")
        if args[2] == "training":
            return lambda train: F.dropout(args[0], args[1], train[0], args[3])
        else:
            return self.call_function(F.dropout, args)

    def apply_view(self, arg1, arg2):
        return lambda input_func: input_func.view(arg1, arg2)
