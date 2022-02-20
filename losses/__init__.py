from .simple_loss import mse, mae, binary_crossentropy

loss_by_name = {
    "mse": mse(),
    "mae": mae(),
    "binary_crossentropy": binary_crossentropy(from_logits=False)
}


def get_loss_by_name(loss):
    if isinstance(loss, type):
        return loss()
    elif isinstance(loss, str):
        if loss in loss_by_name.keys():
            return loss_by_name[loss]
        else:
            raise Exception("cannot find losses %s" % loss)
    else:
        return loss

