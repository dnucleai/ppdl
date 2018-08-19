import argparse
import math

import torch
from torchvision import datasets, transforms
import torch.optim as optim

from ppdl.test.simple.neural_network import Net
from ppdl.client.ppdl_client import PpdlClient
from ppdl.client.http_parameter_manager import HttpParameterManager

THETA = 0.01


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--train", type=int, choices=[0, 1])
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    training_data = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    print("Selecting training data:")

    length = len(training_data)

    if args.train == 0:
        training_data.train_data = training_data.train_data[0:math.floor(length / 2)]
        training_data.train_labels = training_data.train_labels[0:math.floor(length / 2)]
    else:
        training_data.train_data = training_data.train_data[math.floor(length / 2):]
        training_data.train_labels = training_data.train_labels[math.floor(length / 2):]

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    server = "http://127.0.0.1:5000"
    register_url = "%s/register" % server
    get_url = "%s/fetch_weights" % server
    post_url = "%s/store_weights" % server

    parameter_manager = HttpParameterManager(register_url, get_url, post_url)
    client = PpdlClient(model, parameter_manager, THETA)
    print("Training:")
    client.train(args, device, train_loader, test_loader, optimizer)


if __name__ == '__main__':
    main()
