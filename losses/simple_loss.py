import numpy as np
from activations import sigmoid

epsilon = 2e-07

class _loss(object):

    def forward(self, y_true, y_pred):
        raise NotImplemented

    def derivative(self, y_true, y_pred):
        raise NotImplemented

class mse(_loss):
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        n = max(y_pred.shape)
        return (1. / (2 * n)) * np.sum((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true


class mae(_loss):
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        n = max(y_pred.shape)
        return (1. / (2 * n)) * np.sum(np.abs(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / np.sqrt((y_pred - y_true + epsilon) ** 2)


class binary_crossentropy(_loss):
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def forward(self, y_true, y_pred):
        N = max(y_true.shape)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        loss_list = []
        for i in range(N):
            y_t = y_true[i]
            y_p = y_pred[i]
            if self.from_logits:
                loss = y_t * np.log(sigmoid(y_p).forward()) + (1 - y_t) * np.log(sigmoid(1 - y_p).forward())
            else:
                y_p = max(min(y_p, 1 - epsilon), epsilon)
                loss = y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)
            loss_list.append(loss)

        return -np.mean(np.array(loss_list))

    def derivative(self, y_true, y_pred):
        if self.from_logits:
            return (sigmoid(y_pred).forward() - y_true) / (sigmoid(y_pred).forward() * (1 - sigmoid(y_pred).forward()))
        else:
            return (y_pred - y_true) / (y_pred * (1 - y_pred))