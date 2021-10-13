"""
Tham khảo: https://github.com/ral99/SGDForLinearModels/blob/master/pysgd/linear_models.py
https://github.com/PR0Grammar/linear_regression/blob/master/multi_variable_plot.py
"""
import numpy as np

class _logistic_regression(object):
    def __init__(self, learning_rate=None, batch_size=None, n_epochs=1000, activation=None, lamda_regular=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lamda_regular = lamda_regular
        if activation is not None:
            self.activation = activation
        else:
            self.activation = self._linear_activation

    def fit(self, X, y):
        raise NotImplemented

    def predict(self, X):
        raise  NotImplemented

    def _linear_activation(self, x):
        return x

class Gradient_Logistic_Regression(_logistic_regression):
    """
        Công thức Gradient Descent để tính Linear Regression

        Thuật toán:
            Tìm giá trị min của hàm mất mát (loss function) theo phương pháp đạo hàm

        Cách tính:
            B1: Khởi tạo w ngẫu nhiên, với learning_rate nhỏ
            B2: w = w - learning_rate * f'(w)
            B3: Nếu f(x) chưa đủ nhỏ, thực hiện lại B2

        Regularization:
            B2: w = w * (1 - alpha * (lamda / m)) - learning_rate * f'(w)

        Cách sử dụng:
            Sử dụng khi bài toán có quá nhiều đặc trưng khiến việc tính toán bằng đại số tuyển tính là không thể.

        Tham khảo:
            https://www.youtube.com/watch?v=YRiCsLajSHI&list=PLDpRz2wA0qZzTcDLeXP5PSCfmQ96l9-Qr&index=18
    """
    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.reshape(-1, 1)

        self.w = np.random.random((X.shape[1], 1))
        _batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
        for i in range(self.n_epochs):
            for j in range(int(X.shape[0] / _batch_size)):
                learning_rate = self.learning_rate if isinstance(self.learning_rate, float) \
                    else self.learning_rate(i * (X.shape[0] / _batch_size) + j)
                sample  = np.random.choice(X.shape[0], _batch_size, replace=False)
                num_feature = X[sample, :].shape[0]
                self.w = self._gradient_descent(self.w, X[sample, :], y[sample, :], learning_rate, num_feature)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        weight = np.dot(X, self.w)
        y_pred = self.activation(weight)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        y_pred_class = np.array(y_pred_class)
        return y_pred, y_pred_class

    def _gradient_descent(sefl, w, X, y, lr, num_feature, lamda_regular=None):
        grad = sefl._grad(w, X, y)
        if lamda_regular is not None:
            r = 1 - ((lr * lamda_regular) / num_feature)
            w = r.T * w - lr * grad
        else:
            w -= lr * grad
        return w

    def _grad(self, w, x, y):
        weight = np.dot(x, w)
        y_pred = self.activation(weight)
        return np.dot(x.T, (y_pred - y))

