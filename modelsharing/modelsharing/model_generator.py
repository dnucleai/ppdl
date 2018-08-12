import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import model_wrapper as MW
import model_pb2 as mpb2

test_file = "test_model"
def main():
    parser = argparse.ArgumentParser(description="Model Parameters")
    parser.add_argument('--tensor', nargs="+", action="append", type=float)
    # parser.add_argument('-d', nargs="+", action="append", type=int)
    args = parser.parse_args()

    model = read_proto_model( test_file )
    m_wrapper = MW.ModelWrapper( model );

    input = torch.randn(1, 1, 32, 32)
    out = m_wrapper.apply(input)
    print(out)


def read_proto_model( file ):
    model = mpb2.Model();
    f = open( file, "rb" )
    model.ParseFromString( f.read() )
    return model


if __name__ == '__main__':
    main()


