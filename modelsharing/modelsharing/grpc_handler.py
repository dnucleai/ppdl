import model_pb2 as mpb2
import weights_pb2 as wpb2
import torch


def read_proto_model(file):
    model = mpb2.Model()
    f = open(file, "rb")
    model.ParseFromString(f.read())
    return model


def convert_tensor_to_grpc(weights_tensor):
    weight_request = wpb2.Weights()
    grpc_list = weight_request.list.add()
    recursive_grpc_list(weights_tensor, grpc_list)
    return grpc_list


def recursive_grpc_list(inputs, grpc_list):
    next_list = grpc_list.list.add()
    if len(inputs.size()) > 1:
        for each in inputs:
            recursive_grpc_list(each, next_list)
    else:
        for each in inputs:
            grpc_list.contents.extend([each[0]])


def convert_grpc_to_tensor(weights):
    weight_list = recursive_list(weights)
    return torch.Tensor(weight_list)


def recursive_list(weights):
    if len(weights.contents) == 0:
        result_list = []
        for each in weights.list:
            result_list.append(recursive_list(each))
        return result_list
    else:
        return weights.contents
