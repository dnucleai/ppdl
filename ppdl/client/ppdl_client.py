import uuid

import torch
import torch.nn.functional as F
from ppdl.utils.torch_utils import flatten_parameters


class PpdlClient:

    def __init__(self, model, parameter_manager, theta):
        self.model = model
        self.parameter_manager = parameter_manager
        self.client_id = str(uuid.uuid1())
        self.theta = theta

    def train_epoch(self, args, device, train_loader, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def train(self, args, device, train_loader, test_loader, optimizer):
        self.parameter_manager.register_client(self.client_id)
        prev = None
        for epoch in range(1, args.epochs + 1):
            self.train_epoch(args, device, train_loader, optimizer, epoch)
            flat_params = flatten_parameters(self.model.parameters())
            if prev is None:
                deltas = flat_params
            else:
                deltas = flat_params - prev
            prev = flat_params
            self.test(args, device, test_loader)
            self.upload_deltas(deltas)
            indices, selected_weights = self.parameter_manager.download_params(self.client_id)
            self.update_params(indices, selected_weights)

    def test(self, args, device, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def update_params(self, indices, selected_weights):
        all_params = flatten_parameters(self.model.parameters())
        all_params[indices] = selected_weights

        start_index = 0
        for param in self.model.parameters():
            flat_param = param.view(-1)
            flat_param = all_params[start_index:start_index + len(flat_param)]
            start_index += len(flat_param)

    def upload_deltas(self, deltas):
        threshold = round(self.theta * len(deltas))
        _, indices = torch.topk(deltas, threshold)
        indices = indices.view(-1)
        new_deltas = torch.zeros(len(deltas))
        new_deltas[indices] = deltas[indices]
        self.parameter_manager.upload_deltas(self.client_id, new_deltas)
