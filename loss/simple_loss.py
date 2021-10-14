import numpy as np

class _loss(object):

    def forward(self):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class mse(_loss):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def forward(self):
        n = self.y_pred.shape[1]
        return (1. / (2 * n)) * np.sum((self.y_true - self.y_pred) ** 2)

    def derivative(self):
        return self.y_pred - self.y_true