#In the future, this class should be used to validate function inputs
class FunctionService():
    def __init__(self, fm):
        self.input = None
        self.training = None
        self.flags = None
        self.fm = fm

    def set_input(self, input):
        self.input = input
        self.fm.set_input(input)

    def set_training(self, training):
        self.training = training

    def set_function(self, func):
        self.function = func

    def clear_inputs(self):
        self.input = None
        self.training = None
        self.flags = None

    def apply_torch_function( self, function_message, training, input=None):
        function_message = function_message
        self.set_input(input)
        self.set_training(training)

        input_function = self.fm.get_function(function_message.function)
        init_args, flags = self.fm.convert_args(function_message.init_arg)
        converted_flags = self.convert_flags(flags)

        init_function = self.fm.apply_function(input_function, init_args)

        if len(converted_flags) >= 1:
            init_function = init_function(converted_flags)

        args = function_message.inp
        args, flags = self.fm.convert_args(args)
        converted_flags = self.convert_flags(flags)

        if input is not None:
            next_args = [input]
            next_args.extend(args)
            args = next_args
        func = self.fm.apply_function(init_function, args)

        if len(converted_flags) >= 1:
            func = func(converted_flags)
        return func

    def convert_flags(self, flags):
        converted_flags = []
        for flag in flags:
            flag_map = {
                'training': self.training
            }
            converted_flags.append(flag_map[flag])
        return converted_flags


