import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def mse(y_true, y_pred):
    return np.mean(np.subtract(y_true, y_pred)**2)

def mae(y_true, y_pred):
   return np.mean(np.abs(np.subtract(y_true, y_pred)))

def r2(y_true, y_pred):
    return 1 - ((np.sum(np.square(y_true - y_pred))) / (np.sum(np.square(y_true - np.mean(y_true)))))
