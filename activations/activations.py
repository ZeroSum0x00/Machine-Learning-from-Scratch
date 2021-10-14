import numpy as np


class _activation(object):

    def forward(self):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class linear(_activation):
    def __init__(self, x):
        self.x = x

    def forward(self):
        return self.x

    def derivative(self):
        return 1


class sigmoid(_activation):
    def __init__(self, x):
        self.x = x

    def forward(self):
        return 1.0 / (1 + np.exp(-self.x))

    def derivative(self):
        return self.forward() * (1 - self.forward())


class tanh(_activation):
    def __init__(self, x):
        self.x = x

    def forward(self):
        return (np.exp(self.x) - np.exp(-self.x)) / (np.exp(self.x) + np.exp(-self.x))

    def derivative(self):
        return 1 - self.forward() ** 2


class relu(_activation):
    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.where(self.x >= 0, self.x, 0)

    def derivative(self):
        return np.where(self.x >= 0, 1, 0)


class leaky_relu(_activation):
    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.where(self.x >= 0, self.x, 0.01 * self.x)

    def derivative(self):
        return np.where(self.x >= 0, 1, 0.01)
