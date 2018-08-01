import uuid

import torch
import torch.nn.functional as F


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
            flat_params = self.flatten_parameters()
            if prev:
                deltas = flat_params - prev
            else:
                deltas = flat_params
            self.upload_deltas(deltas)
            new_deltas = self.parameter_manager.download_params(flat_params)
            self.update_params(new_deltas)

    def update_params(self, new_deltas):
        start_index = 0
        for param in self.model.parameters():
            flat_param = param.view(-1)
            flat_param += new_deltas[start_index:start_index + flat_param.size()]
            start_index += flat_param.size()

    def flatten_parameters(self):
        return torch.cat((p.data.view(-1) for p in self.model.parameters()))

    def upload_deltas(self, deltas):
        threshold = round(self.theta * deltas.size())
        _, indices = torch.topk(deltas, threshold)
        new_deltas = torch.zeros(deltas.size())
        new_deltas[indices] = deltas[indices]
        self.parameter_manager.upload_deltas(new_deltas, self.client_id)
