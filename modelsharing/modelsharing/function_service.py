import torch
import torch.nn as nn
import torch.nn.functional as F
import model_pb2 as mpb2

def apply_torch_function( function_message, input=None):
    print(function_message.function)

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
    args = convert_args(args)
    if len(args) == 3:
        return func(args[0], args[1], args[2])
    elif len(args) == 2:
        return func(args[0], args[1])
    elif len(args) == 1:
        return func(args[0])
    else:
        return func

def convert_args(args):
    converted_args = []
    for arg in args:
        if represents_int(arg):
            converted_args.append(int(arg))
        else:
            converted_args.append(arg)
    return converted_args

def represents_int(s):
    try:
        val = int(s)
        return True
    except (ValueError, TypeError):
        return False

def view( input ):
    return input.view

function_map = {
    'Conv2d': nn.Conv2d,
    'Dropout2d': nn.Dropout2d,
    'max_pool2d': F.max_pool2d,
    'relu': F.relu,
    'view': view,
    'Linear': nn.Linear,
    'dropout': F.dropout,
    'log_softmax': F.log_softmax
}