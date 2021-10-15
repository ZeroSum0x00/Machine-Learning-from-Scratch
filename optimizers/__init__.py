from .sgd import SGD

optimizer_by_name = {
    "SGD": SGD(),
}


def get_optimizer_by_name(optimizer):
    if isinstance(optimizer, type):
        return optimizer()
    elif isinstance(optimizer, str):
        if optimizer in optimizer_by_name.keys():
            return optimizer_by_name[optimizer]
        else:
            raise Exception("cannot find optimizer %s" % optimizer)
    else:
        return optimizer

