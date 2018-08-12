import sys
import model_pb2 as mpb2

def main():
    model = mpb2.Model()
    model.type = "cnn"

    conv1 = model.function.add()
    max_pool1 = model.function.add()
    relu1 = model.function.add()
    conv2 = model.function.add()
    conv2_drop = model.function.add()
    max_pool2 = model.function.add()
    relu2 = model.function.add()
    view = model.function.add()
    fc1 = model.function.add()
    relu3 = model.function.add()
    dropout = model.function.add()
    fc2 = model.function.add()
    log_softmax = model.function.add()

    conv1.function = "Conv2d"
    conv1.init_arg.extend(["1", "10", "5"])

    max_pool1.function = "max_pool2d"
    max_pool1.inp.extend(["2"])

    relu1.function = "relu"

    conv2.function = "Conv2d"
    conv2.init_arg.extend(["10", "20", "5"]);

    conv2_drop.function = "Dropout2d"
    conv2_drop.init_arg.extend([""])

    max_pool2.function = "max_pool2d"
    max_pool2.inp.extend(["2"])

    relu2.function = "relu"

    view.function = "view"
    view.init_arg.extend(["-1", "num_flat_features"])

    fc1.function = "Linear"
    fc1.init_arg.extend(["num_flat_features", "50"])

    relu3.function = "relu"

    dropout.function = "dropout"
    dropout.inp.extend(["self.training"])

    fc2.function = "Linear"
    fc2.init_arg.extend(["50", "10"])

    log_softmax.function = "log_softmax"
    log_softmax.inp.extend(["1"])

    # Write the new model to disk.
    f = open(sys.argv[1], "wb+")
    f.write(model.SerializeToString())
    f.close()



if __name__ == '__main__':
    main()