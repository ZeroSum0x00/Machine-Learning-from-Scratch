from .activations import linear, sigmoid, tanh, relu, leaky_relu

loss_by_name = {
    "linear": linear,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu
}


def get_activation_by_name(activation):
    if isinstance(activation, type):
        return activation()
    elif isinstance(activation, str):
        if activation in loss_by_name.keys():
            return loss_by_name[activation]
        else:
            raise Exception("cannot find activations %s" % activation)
    else:
        return activation

