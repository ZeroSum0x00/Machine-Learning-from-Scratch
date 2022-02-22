import itertools

import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, y_true, y_pred, visual=False):
        self.tp = true_positive(y_true, y_pred)
        self.fp = false_positive(y_true, y_pred)
        self.tn = true_negative(y_true, y_pred)
        self.fn = false_negative(y_true, y_pred)

        if visual:
            cm = np.array([[self.tp, self.fn],
                           [self.fp, self.tn]])
            classes = np.unique(y_true)
            self.__visual(cm, classes)

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

    def __call__(self):
        return np.array([[self.tp, self.fn],
                         [self.fp, self.tn]])

    def __visual(self, cm, classes):
        # Tham khảo tại: https://machinelearningcoban.com/2017/08/31/evaluation/
        figsize = plt.rcParams.get('figure.figsize')
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Binary Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

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
