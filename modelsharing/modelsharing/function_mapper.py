import torch
import torch.nn as nn
import torch.nn.functional as F
import model_pb2 as mpb2
import function_service as fs


f_service = fs.FunctionService(None)

def apply_torch_function( function_message, training, input=None):
    f_service.set_input(input)
    f_service.set_training(training)

    func = function_map[function_message.function]
    init_args = function_message.init_arg
    func = apply_function_on_args(func, init_args)
    args = function_message.inp
    if input is not None:
        next_args = [input]
        next_args.extend(args)
        args = next_args
    func = apply_function_on_args(func, args)
    return func


def apply_function_on_args(func, args):
    print(func)
    args = convert_args(args)
    try:
        result = call_function(func, args)
        return result
    except (RuntimeError):
        raise ValueError("Function specified by message does not exist or has an incorrect number of arguments.")


def call_function(func, args):
    if len(args) == 4:
        return func(args[0], args[1], args[2], args[3])
    elif len(args) == 3:
        return func(args[0], args[1], args[2])
    elif len(args) == 2:
        return func(args[0], args[1])
    elif len(args) == 1:
        if is_empty(args[0]):
            return func()
        return func(args[0])
    else:
        return func

def convert_args(args):
    converted_args = []
    for arg in args:
        if represents_int(arg):
            converted_args.append(int(arg))
        elif in_function_map(arg):
            converted_args.append(function_map[arg]())
        elif represents_class(arg):
            converted_args.append(arg)
        elif is_empty(arg):
            converted_args.append(arg)
        elif in_string_set(arg):
            converted_args.append(arg)
        else:
            raise ValueError("Argument for called function is not properly specified, or is not yet implemented.")
    return converted_args

def represents_int(s):
    try:
        val = int(s)
        return True
    except (ValueError, TypeError):
        return False

def represents_class(cl):
    for c in class_map.values():
        if isinstance(cl, c):
            return True
    return False

def is_empty(arg):
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

def in_function_map(input):
    return input in function_map.keys()

def in_string_set(arg):
    return arg in string_set

def num_flat_features():
    f = f_service.num_flat_features()
    return f

def apply_dropout(*args):
    return f_service.apply_dropout(args)

def apply_view( arg1, arg2 ):
    return f_service.apply_view(arg1, arg2)

function_map = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Dropout2d': nn.Dropout2d,
    'max_pool2d': F.max_pool2d,
    'relu': F.relu,
    'view': apply_view,
    'Linear': nn.Linear,
    'dropout': apply_dropout,
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
    'num_flat_features': num_flat_features,
}

class_map = {
    'tensor': torch.Tensor,
    'Conv1d': nn.Conv1d,
    'Dropout2d': nn.Dropout2d,
    'Linear': nn.Linear
}

string_set = set([
    'training'
])