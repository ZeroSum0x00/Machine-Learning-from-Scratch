import numpy as np
from tensorflow.keras.layers import Activation

class Linear(Activation):
    def __init__(self, **kwargs):
        super(Linear, self).__init__(lambda x : np.where(x >= 0, 1, 0), **kwargs)

class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(lambda x : 1.0 / (1 + np.exp(-x)), **kwargs)
