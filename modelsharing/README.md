To create a test model with specific inputs, create a file with imports:

import model_pb2 as mpb2

Then define a function and run:
    model = mpb2.Model()
    model.type = "cnn"

Then for each function you want the forward step of the model to have, you need to run (in the order the functions should appear):
func = model.function.add()

Finally, for each function you need to specify the function name (as listed in the function map)
and specify (if any) its INIT arguments, and its EXECUTION arguments.

Examples:
conv1.function = "Conv2d" #Function name
conv1.init_arg.extend(["1", "10", "5"]) #Init args, used to specify that the function/class should be initialized with arguments

conv2_drop.function = "Dropout2d"
conv2_drop.init_arg.extend([""])  #Empty init argument used to specify that the function/class should be initialized without any arguments

dropout.function = "dropout"
dropout.inp.extend(["0.5","training", "False"]) #Input arguments and no init arguments mean
                                                #that the function will not be initialized, and will run with the specified arguments

view.function = "view"
view.init_arg.extend(["-1", "num_flat_features"]) #Note that this method is initialized using a function as an argument.
                                                  #The function 'num_flat_features' itself was defined in the function mapper
                                                  #Only functions defined in the function mapper can be passed in like this

Note that when a method runs, it ALWAYS takes the result of the previous function as input. In the cases where the function itself
requires a different formulation (e.g. x.view(-1, num_flat_features(x)) ), the function was redefined in the function mapper so
that this specification is allowed.

Once a model is defined, simply write it to a file, and call 'python model_generator.py {file}'
