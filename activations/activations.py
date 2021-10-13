import numpy as np

def linear(x):
    return x

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.where(x >= 0, x, 0)

def leaky_relu(x):
    return np.where(x >= 0, x, 0.01 * x)

def swish(x):
    return x / (1 + np.exp(-x))
