import torch.nn.functional as F

#In the future, this class should be used to validate function inputs
class FunctionService():
    def __init__(self, input, training=None):
        self.input_function = input
        self.training = training

    def set_input(self, input):
        self.input_function = input

    def set_training(self, training):
        self.training = training

    def apply_view(self, arg1, arg2):
        return lambda x: self.input_function.view(arg1, arg2)

    def apply_dropout(self, *args):
        if len(args) == 1:
            if args[0] == 'training':
                return F.dropout(self.input_function, training=self.training)
            else:
                return F.dropout(self.input_function)

        elif len(args) == 2:
            if args[0] == 'training':
                return F.dropout(self.input_function, args[1], training=self.training)
            else:
                return F.dropout(self.input_function, args[1])

        elif len(args) == 3:
            if args[0] == 'training':
                return F.dropout(self.input_function, args[1], args[0], args[2])
            else:
                return F.dropout(self.input_function, args[1], inplace=args[2])

    def num_flat_features(self):
        if self.input_function is None:
            return
        size = self.input_function.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

