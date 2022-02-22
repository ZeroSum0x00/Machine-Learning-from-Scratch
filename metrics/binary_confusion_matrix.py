import numpy as np

def true_positive(y_true, y_pred):
    positive = np.unique(y_true)[1]
    return np.sum(y_pred[y_true == y_pred] == positive)

def false_positive(y_true, y_pred):
    positive = np.unique(y_true)[1]
    return np.sum(y_pred[y_true != y_pred] == positive)

def true_negative(y_true, y_pred):
    negative = np.unique(y_true)[0]
    return np.sum(y_pred[y_true == y_pred] == negative)

def false_negative(y_true, y_pred):
    negative = np.unique(y_true)[0]
    return np.sum(y_pred[y_true != y_pred] == negative)

class binary_confusion_matrix:
    def __init__(self, y_true, y_pred):
        self.tp = true_positive(y_true, y_pred)
        self.fp = false_positive(y_true, y_pred)
        self.tn = true_negative(y_true, y_pred)
        self.fn = false_negative(y_true, y_pred)

    def __str__(self):
        return """
                         |  Predicted  |  Predicted  |
                         | as Positive | as Negative |
        -----------------|-------------|-------------|
        Actual: Positive | {:<12}| {:<12}|
        -----------------|-------------|-------------|
        Actual: Negative | {:<12}| {:<12}|
        -----------------|-------------|-------------|
        """.format(self.tp, self.fn, self.fp, self.tn)

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def f_score(y_true, y_pred, beta=1):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return ((1 + beta**2) * p * r) / (beta**2 * (p + r))
