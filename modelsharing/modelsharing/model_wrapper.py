import models.cnn_model as cnn
class ModelWrapper:

    def __init__(self, request):
        self.create_model_from_proto( request )

    def getType (self, request):
        return request.type

    def create_model_from_proto(self, request):
        model_type = self.getType(request)
        model = {
            'cnn': cnn.CNNModel
        }[model_type]
        self.model = model(request)

    def create_model_from_args(self, args):
        self.model = {
            'cnn': cnn.CNNModel
        }[args]
        self.model(args)

    def forward(self, x):
        self.model.forward(x)

    def train(self, args, device, train_loader, optimizer, epoch):
        self.model.train(args, device, train_loader, optimizer, epoch)

    def check_model_is_nn(self):
        return self.model.is_nn()


