"""
Tham khảo: https://github.com/ral99/SGDForLinearModels/blob/master/pysgd/linear_models.py
https://github.com/PR0Grammar/linear_regression/blob/master/multi_variable_plot.py
"""
import numpy as np
from abc import ABC, abstractmethod

class _linear_regression(ABC):

    def __init__(self, learning_rate=None, batch_size=None, n_epochs=1000, lamda_regular=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lamda_regular = lamda_regular

    @abstractmethod
    def fit(self, X, y):
        raise NotImplemented

    @abstractmethod
    def predict(self, X):
        raise  NotImplemented

class Normal_Equation_Linear_Regression(_linear_regression):
    """
        Công thức Normal Equation để tính Linear Regression

        Cách tính:
            w = (X.T * X)^(-1) * X.T * y

        Regularization:
                                     _           _
                                    | 1          |
            w = (X.T * X + lambda * |   1        |)^(-1) * X.T * y
                                    |     ...    |
                                    |         1  |
                                    -           -

        Cách sử dụng:
            Chỉ sử dụng Đại số tuyến tính để giải, không sử dụng Gradient Descent (learning rate, iterate)
            nên thuật toán nhanh khi bài toán có số đặc trưng nhỏ hơn 10000, độ phức tạp của thuật toán là
            o(n^3) trong đó n là số chiều của ma trận X

        Tham khảo:
            https://www.youtube.com/watch?v=YRiCsLajSHI&list=PLDpRz2wA0qZzTcDLeXP5PSCfmQ96l9-Qr&index=18
            https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/
    """
    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.reshape(-1, 1)
        _batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
        for batch in range(int(X.shape[0] / _batch_size)):
            self.w = self._normal_equation(X, y, self.lamda_regular)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(self.w.T, X.T).flatten()

    def _normal_equation(self, X, y, lamda_regular=None):
        if lamda_regular is not None:
            # print('Sử dụng Regularization')
            ident_matrix = np.eye(X.shape[1])
            ident_matrix[0, 0] = 0
            A = np.dot(X.T, X) + lamda_regular * ident_matrix
        else:
            A = np.dot(X.T, X)

        try:
            A_pinv = np.linalg.inv(A)
        except:
            # print('A là ma trận không khả nghịch')
            A_pinv = np.linalg.pinv(A)

        B = np.dot(X.T, y)
        return np.dot(A_pinv, B)

class Gradient_Linear_Regression(_linear_regression):
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
        return np.dot(np.transpose(self.w), np.transpose(X)).flatten()

    def _gradient_descent(sefl, w, X, y, lr, num_feature, lamda_regular=None):
        grad = sefl._grad(w, X, y, num_feature)
        if lamda_regular is not None:
            r = 1 - ((lr * lamda_regular) / num_feature)
            w = r.T * w - lr * grad
        else:
            w -= lr * grad
        return w

    def _grad(self, w, X, y, num_feature):
        y_pred = np.dot(X, w) - y
        return (2.0 / num_feature) * np.dot(X.T, y_pred)
