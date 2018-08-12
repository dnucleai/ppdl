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

    def train(self, args, device, train_loader, optimizer):
        self.parameter_manager.register_client(self.client_id)
        prev = None
        for epoch in range(1, args.epochs + 1):
            self.train_epoch(args, device, train_loader, optimizer, epoch)
            flat_params = flatten_parameters(self.model.parameters())
            if prev:
                deltas = flat_params - prev
            else:
                deltas = flat_params
            self.upload_deltas(deltas)
            indices, selected_weights = self.parameter_manager.download_params(flat_params)
            self.update_params(indices, selected_weights)

    def update_params(self, indices, selected_weights):
        all_params = flatten_parameters(self.model.parameters())
        all_params[indices] = selected_weights

        start_index = 0
        for param in self.model.parameters():
            flat_param = param.view(-1)
            flat_param = all_params[start_index:start_index + flat_param.size()]
            start_index += flat_param.size()

    def upload_deltas(self, deltas):
        threshold = round(self.theta * deltas.size())
        _, indices = torch.topk(deltas, threshold)
        indices = indices.view(-1)
        new_deltas = torch.zeros(deltas.size())
        new_deltas[indices] = deltas[indices]
        self.parameter_manager.upload_deltas(new_deltas, self.client_id)
