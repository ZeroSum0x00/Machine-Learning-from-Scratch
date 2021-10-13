import numpy as np

def linear(x):
    return x

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def relu(x):
    return np.where(x >= 0, 1, 0)