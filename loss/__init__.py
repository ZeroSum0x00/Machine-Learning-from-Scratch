from .simple_loss import mse

loss_by_name = {
    "mse": mse,
}


def get_loss_by_name(loss):
    if isinstance(loss, type):
        return loss()
    elif isinstance(loss, str):
        if loss in loss_by_name.keys():
            return loss_by_name[loss]
        else:
            raise Exception("cannot find loss %s" % loss)
    else:
        return loss

